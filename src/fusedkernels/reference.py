"""PyTorch reference implementations (the "eager" baselines and the numerics oracle)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Eager composite RMSNorm, computed in fp32 (the HF Llama/Qwen recipe, but
    multiplying by the weight in fp32 like our kernel). ~6 kernel launches."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
    return y.to(x.dtype)


def add_rmsnorm_ref(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    h = x + residual  # materialized in the storage dtype, exactly like an unfused model
    return rmsnorm_ref(h, weight, eps), h


def rmsnorm_native(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """PyTorch's built-in F.rms_norm (a single fused ATen kernel on CUDA in recent releases)."""
    return F.rms_norm(x, (x.shape[-1],), weight, eps)


def attention_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Textbook causal attention that materializes the N x N matrix, in fp32.
    Memory: B*H*N^2*4 bytes -- only use for small N."""
    n, d = q.shape[-2], q.shape[-1]
    s = (q.float() @ k.float().transpose(-1, -2)) / d**0.5
    mask = torch.ones(n, n, dtype=torch.bool, device=q.device).tril()
    s = s.masked_fill(~mask, float("-inf"))
    return (torch.softmax(s, dim=-1) @ v.float()).to(q.dtype)


def attention_eager(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Same math in the input dtype (what a naive model implementation runs)."""
    n, d = q.shape[-2], q.shape[-1]
    s = (q @ k.transpose(-1, -2)) * d**-0.5
    mask = torch.ones(n, n, dtype=torch.bool, device=q.device).tril()
    s = s.masked_fill(~mask, float("-inf"))
    return torch.softmax(s.float(), dim=-1).to(q.dtype) @ v


def attention_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """PyTorch SDPA; on this GPU it dispatches to its bundled FlashAttention-2 kernel."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)
