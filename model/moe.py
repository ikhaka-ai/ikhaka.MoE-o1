#This is the code that makes this an MoE rather than a dense transformer
#Motivation: Routing is difficult and supervised. The domain expert will pick it's expert. There
#is no router network, no learned gate, no top-k softmax over experts. Every token in a given sequence
#goes through the same expert.

#This allows 3 things that a learned router does not have (1) Static shapes, (2) no router collapse, (3) a distinct expert for a particular expert

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MoEConfig

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x))*self.w_up(x))

class DemixMoE(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.experts = nn.ModuleList(SwiGLU(cfg.d_model, cfg.d_ff) for _ in range(cfg.n_experts))
        self.register_buffer("_token_counts",torch.zeros(cfg.n_experts, dtype=torch.long),persistent=False,)

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        assert domain_ids.shape == (B,), (
            f"domain_ids must be one id per sequence, got shape {tuple(domain_ids.shape)} "
            f"for a batch of {B}"
        )
        out = torch.zeros_like(x)
        for expert_id, expert in enumerate(self.experts):
            mask = domain_ids == expert_id
            if not mask.any():
                continue
            out[mask] = expert(x[mask])
            if self.training:
                self._token_counts[expert_id] += int(mask.sum())*T

        return out

    def aux_state(self)->dict:
        return {"token_counts":self._token_counts.clone()}

    def load_aux_state(self, state: dict)->None:
        self._token_counts.copy_(state["token_counts"])

    def utilization(self)->torch.Tensor:
        total = self._token_counts.sum().clamp(min=1)
        return self._token_counts.float()/total
