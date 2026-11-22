"""Fused RMSNorm forward (optionally fused with the residual add) in Triton.

    y = x / sqrt(mean(x^2) + eps) * w                     (Zhang & Sennrich, 2019)

In a pre-norm transformer block the norm is always preceded by a residual add:

    h   = x + residual          # 1 read x, 1 read residual, 1 write h
    y   = rmsnorm(h) * w        # 1 read h, 1 write y

Run as two PyTorch ops this moves 5 tensors through DRAM. Fused into one kernel
it moves 4 (read x, read residual, write h, write y): `h` never makes the round
trip. Without the residual the fused kernel moves exactly 2 tensors (read x,
write y), while a naive eager composite (`x.float()`, `pow`, `mean`, `rsqrt`,
`mul`, `mul`, `.to`) launches ~7 kernels and moves each intermediate through
DRAM. The arithmetic is trivial (~4 FLOP per element), so the
kernel is purely memory bound: the only thing that matters is bytes moved.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# One row must fit in a single program's registers. 16384 covers every
# current LLM hidden size (Llama-3-405B: 16384, Qwen3-0.6B: 1024).
MAX_FUSED_N = 16384


@triton.jit
def _rmsnorm_fwd_kernel(
    X, R, W, Y, R_OUT,  # pointers
    stride_x, stride_r, stride_y, stride_ro,  # row strides (elements)
    N, eps,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # One program per row: the row is loaded once into registers, reduced, and
    # written once. That is the "one read + one write" ideal of a memory-bound op.
    row = tl.program_id(0).to(tl.int64)  # int64: row*stride can overflow int32
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    if HAS_RESIDUAL:
        r = tl.load(R + row * stride_r + cols, mask=mask, other=0.0)
        # Round the sum to the storage dtype *before* normalizing. This matches
        # the unfused PyTorch semantics (h = x + residual is materialized in
        # fp16/bf16) so the fused and unfused models produce the same residual
        # stream bit-for-bit, which matters when it is carried across 30+ layers.
        h = x + r
        tl.store(R_OUT + row * stride_ro + cols, h, mask=mask)
        x = h
    # Accumulate in fp32: sum of N squares in fp16 overflows at |x| > ~256/sqrt(N)
    # and loses precision long before that.
    xf = x.to(tl.float32)
    mean_sq = tl.sum(xf * xf, axis=0) / N
    rstd = tl.rsqrt(mean_sq + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = xf * rstd * w
    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)


def _launch_config(n: int) -> tuple[int, int]:
    block_n = triton.next_power_of_2(n)
    if block_n > MAX_FUSED_N:
        raise ValueError(f"hidden size {n} > {MAX_FUSED_N} does not fit one program")
    # ~8-16 elements per thread keeps register pressure low and gives each warp
    # enough independent 16-byte loads in flight to cover DRAM latency.
    num_warps = min(max(block_n // 512, 1), 16)
    return block_n, num_warps


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm over the last dim.

    Without `residual`: returns y = rmsnorm(x) * weight.
    With `residual`:   returns (y, h) where h = x + residual (the new residual
    stream) and y = rmsnorm(h) * weight -- the vLLM `fused_add_rms_norm` contract,
    except that we write `h` to a fresh tensor instead of in place.
    """
    assert x.is_cuda and x.shape[-1] == weight.shape[0]
    n = x.shape[-1]
    x2 = x.reshape(-1, n)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    m = x2.shape[0]
    y = torch.empty_like(x2)
    block_n, num_warps = _launch_config(n)

    if residual is not None:
        assert residual.shape == x.shape and residual.dtype == x.dtype
        r2 = residual.reshape(-1, n)
        if r2.stride(-1) != 1:
            r2 = r2.contiguous()
        h = torch.empty_like(x2)
        _rmsnorm_fwd_kernel[(m,)](
            x2, r2, weight, y, h,
            x2.stride(0), r2.stride(0), y.stride(0), h.stride(0),
            n, eps, HAS_RESIDUAL=True, BLOCK_N=block_n, num_warps=num_warps,
        )
        return y.view(x.shape), h.view(x.shape)

    # R / R_OUT are dead when HAS_RESIDUAL=False (the branch is compiled out),
    # so any valid pointer works.
    _rmsnorm_fwd_kernel[(m,)](
        x2, x2, weight, y, y,
        x2.stride(0), 0, y.stride(0), 0,
        n, eps, HAS_RESIDUAL=False, BLOCK_N=block_n, num_warps=num_warps,
    )
    return y.view(x.shape)


def rmsnorm_bytes(m: int, n: int, elem_bytes: int, residual: bool) -> int:
    """Minimum DRAM traffic of the fused kernel (the weight vector is negligible
    and stays in L2, but we count it once anyway)."""
    tensors = 4 if residual else 2  # x,(r) read + y,(h) write
    return tensors * m * n * elem_bytes + n * elem_bytes


def rmsnorm_flops(m: int, n: int, residual: bool) -> int:
    """x*x, +=, *rstd, *w  (+1 add for the residual). The rsqrt per row is noise."""
    return (5 if residual else 4) * m * n
