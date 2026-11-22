"""Fused Triton kernels (RMSNorm, causal FlashAttention-2 forward) + roofline tools."""

from .attention import attention_bytes, attention_flops, causal_attention
from .rmsnorm import rmsnorm, rmsnorm_bytes, rmsnorm_flops

__all__ = [
    "attention_bytes",
    "attention_flops",
    "causal_attention",
    "rmsnorm",
    "rmsnorm_bytes",
    "rmsnorm_flops",
]
