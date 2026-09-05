#shared trunk. Every layer in this file runs identically regardless of the domain
#a token belongs to. This enables the four experts to work together later on.
#They all read and write to the same residual stream shaped by the same attention.

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MoEConfig

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight= nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(dim=-1,keepdim=True)+self.eps)

        return (x*rms).to(dtype)*self.weight

def precompute_rope(head_dim: int, seq_len: int, theta: float = 10_000.0,device=None)->tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0/(theta**(torch.arange(0, head_dim,2,device=device).float()/head_dim))
    t = torch.arange(seq_len,device=device).float()
    freqs = torch.outer(t, inv_freq)
    freqs = torch.cat([freqs, freqs], dim=-1)
    return freqs.cos(), freqs.sin()

def _rotate_half(x: torch.Tensor)->torch.Tensor:
    d = x.shape[-1]//2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([-x2, x1], dims=-1)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor)->torch.Tensor:
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + _rotate_half(x) * sin

