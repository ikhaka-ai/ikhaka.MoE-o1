Try AI directly in your favourite apps … Use Gemini to generate drafts and refine content, plus get Gemini Pro with access to Google's next-gen AI

"""
this code wires the model/layer interfaces (shared trunk) to the model(mixture of
experts into the full model.

each sublayer normalise it's input and add it's output to the residual stream, rather
than normalising the output.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import MoEConfig
from .layers import GQAAttention, RMSNorm, precompute_rope
from .moe import DemixMoE

class Block(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.attn = GQAAttention(cfg)
        self.moe_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.moe = DemixMoE(cfg)

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.moe(self.moe_norm(x), domain_ids)
        return x

#model
class MoETransformer(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)

        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        cos, sin = precompute_rope(cfg.head_dim, cfg.seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def forward(self, input_ids: torch.Tensor, domain_ids: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = input_ids.shape
        assert T <= self.cfg.seq_len, (
            f"sequence length {T} exceeds the {self.cfg.seq_len} the RoPE "
            f"table was precomputed for"
        )
        x = self.embed(input_ids)
        cos = self.rope_cos[:T].to(x.device)
        sin = self.rope_sin[:T].to(x.device)

        for block in self.blocks:
            x = block(x, domain_ids, cos, sin)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    def num_params(self, active_only: bool = False) -> int:
        """
        active_only=False -> total parameters (what sits in memory)
        active_only=True  -> parameters touched per token (what compute scales with)

        Cross-check this against model_numbers.py from the design document --
        they should agree to within rounding.
        """
        if not active_only:
            return sum(p.numel() for p in self.parameters())

        total = sum(p.numel() for p in self.parameters())
        moe_total = sum(
            p.numel() for block in self.blocks for p in block.moe.parameters()
        )
        moe_active = moe_total // self.cfg.num_experts
        return total - moe_total + moe_active

    def aux_state(self) -> dict:
        """For checkpointing.py: expert load counters, one dict per layer."""
        return {i: block.moe.aux_state() for i, block in enumerate(self.blocks)}

    def load_aux_state(self, state: dict) -> None:
        for i, block in enumerate(self.blocks):
            block.moe.load_aux_state(state[i])
