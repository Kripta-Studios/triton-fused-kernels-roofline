"""Causal attention forward, FlashAttention-2 style, in Triton.

    O = softmax(Q K^T * scale + causal_mask) V        Q,K,V,O: [B, H, N, D]

Standard attention writes the N x N score matrix S and probability matrix P to
DRAM: O(N^2) bytes. FlashAttention never materializes them. Each program owns
a BLOCK_M tile of query rows, keeps it in registers/SRAM, and streams K/V tiles
past it. The softmax is computed *online* (Milakov & Gimelshein, 2018): we keep
per-row running max `m` and running denominator `l`, and rescale the partial
output whenever the max grows:

    m_new = max(m, rowmax(S_j))
    alpha = exp(m - m_new)              # correction for everything seen so far
    P_j   = exp(S_j - m_new)
    l     = alpha * l + rowsum(P_j)
    acc   = alpha * acc + P_j @ V_j
    ...after the last tile:  O = acc / l

FA2-specific choices visible below (Dao, 2023):
  * the outer loop is over query tiles (one per program), the inner over K/V,
    so no inter-program communication is needed;
  * the division by `l` is deferred to the very end (one divide per row, not
    one per tile);
  * the causal mask is only evaluated on the diagonal tiles; the tiles strictly
    below the diagonal run a mask-free fast path, and tiles above it are skipped
    entirely (half the FLOPs of non-causal attention).
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634  # exp(x) == exp2(x * log2(e)); exp2 is the native SFU op


@triton.jit
def _attn_inner(
    acc, l_i, m_i, q,
    k_ptrs, v_ptrs, stride_kn, stride_vn,
    offs_m, offs_n, lo, hi, N_CTX, qk_scale,
    BLOCK_N: tl.constexpr, DIAGONAL: tl.constexpr,
):
    """Stream K/V tiles [lo, hi) past the resident Q tile, updating (acc, l, m)."""
    k_ptrs += lo * stride_kn
    v_ptrs += lo * stride_vn
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        if DIAGONAL:
            # Only the diagonal tile can run past the end of the sequence.
            kv_mask = (start_n + offs_n) < N_CTX
            k = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0)
        else:
            k = tl.load(k_ptrs)
        # S_j = Q K_j^T on the tensor cores, fp32 accumulate. `qk_scale` folds the
        # 1/sqrt(D) softmax scale and log2(e) together so we can use exp2.
        s = tl.dot(q, tl.trans(k)) * qk_scale
        if DIAGONAL:
            causal = offs_m[:, None] >= (start_n + offs_n[None, :])
            s = tl.where(causal, s, float("-inf"))
        # --- online softmax update (see module docstring) ---
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.math.exp2(s - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        if DIAGONAL:
            v = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0)
        else:
            v = tl.load(v_ptrs)
        # P is cast to the input dtype so the second GEMM also runs on fp16/bf16
        # tensor cores (same choice as FA2); the accumulator stays fp32.
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_new
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
    return acc, l_i, m_i


# A deliberately small search space: every config is a legal FA2 tiling on a
# ~100 KB-SMEM consumer GPU. BLOCK_M must be a multiple of BLOCK_N so the
# diagonal of a query tile is covered by whole K/V tiles.
_CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w, num_stages=s)
    for bm, bn, w, s in [
        (64, 32, 4, 3),
        (64, 64, 4, 3),
        (128, 32, 4, 3),
        (128, 64, 8, 2),
        (128, 64, 4, 2),
    ]
]


@triton.autotune(configs=_CONFIGS, key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, O,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    H, N_CTX, qk_scale,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Causal work is triangular: query tile i touches i+1 K/V tiles. Launching
    # the *largest* tiles first (reverse order) lets the scheduler back-fill
    # SMs with the short tiles at the end instead of leaving a long tail.
    start_m = tl.num_programs(0) - 1 - tl.program_id(0)
    off_bh = tl.program_id(1)
    off_z = (off_bh // H).to(tl.int64)
    off_h = (off_bh % H).to(tl.int64)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = (Q + off_z * stride_qz + off_h * stride_qh
              + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    k_ptrs = (K + off_z * stride_kz + off_h * stride_kh
              + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk)
    v_ptrs = (V + off_z * stride_vz + off_h * stride_vh
              + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)

    m_mask = offs_m < N_CTX
    # The Q tile is loaded once and reused against every K/V tile: this reuse
    # is what turns attention from memory-bound into compute-bound.
    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Stage 1: tiles strictly below the diagonal -- no mask, no bounds checks.
    diag_lo = start_m * BLOCK_M
    acc, l_i, m_i = _attn_inner(
        acc, l_i, m_i, q, k_ptrs, v_ptrs, stride_kn, stride_vn,
        offs_m, offs_n, 0, diag_lo, N_CTX, qk_scale,
        BLOCK_N=BLOCK_N, DIAGONAL=False,
    )
    # Stage 2: the BLOCK_M x BLOCK_M diagonal block, masked. Every query row has
    # at least one unmasked key here (itself), so m_new is never -inf and
    # exp2(-inf - -inf) = NaN cannot occur.
    acc, l_i, m_i = _attn_inner(
        acc, l_i, m_i, q, k_ptrs, v_ptrs, stride_kn, stride_vn,
        offs_m, offs_n, diag_lo, tl.minimum(diag_lo + BLOCK_M, N_CTX), N_CTX, qk_scale,
        BLOCK_N=BLOCK_N, DIAGONAL=True,
    )

    acc = acc / l_i[:, None]  # deferred normalization (FA2)
    o_ptrs = (O + off_z * stride_oz + off_h * stride_oh
              + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=m_mask[:, None])


def causal_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float | None = None
) -> torch.Tensor:
    """Causal self-attention forward. q, k, v: [B, H, N, D] fp16/bf16, same shape.

    D must be a power of two in [16, 128] (tl.dot / register-tile constraint).
    GQA/MQA is not handled here; expand K/V heads before calling.
    """
    assert q.shape == k.shape == v.shape and q.dim() == 4
    assert q.dtype in (torch.float16, torch.bfloat16) and q.dtype == k.dtype == v.dtype
    b, h, n, d = q.shape
    assert d in (16, 32, 64, 128), f"head_dim {d} not supported"
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)
    o = torch.empty_like(q)

    def grid(meta: dict) -> tuple[int, int]:
        return (triton.cdiv(n, meta["BLOCK_M"]), b * h)

    _attn_fwd_kernel[grid](
        q, k, v, o,
        *q.stride(), *k.stride(), *v.stride(), *o.stride(),
        h, n, sm_scale * LOG2E,
        HEAD_DIM=d,
    )
    return o


def attention_flops(b: int, h: int, n: int, d: int, causal: bool = True) -> float:
    """Two GEMMs (QK^T and PV), each 2*N*N*D FLOPs per head. Causal skips the
    upper triangle, i.e. half (we use the exact N(N+1)/2 count)."""
    pairs = n * (n + 1) / 2 if causal else n * n
    return 4.0 * b * h * pairs * d


def attention_bytes(b: int, h: int, n: int, d: int, elem_bytes: int) -> int:
    """Ideal (compulsory) DRAM traffic: read Q, K, V once, write O once.
    The real kernel re-reads K/V once per query tile, but those re-reads mostly
    hit the 40 MB L2 for the shapes benchmarked here."""
    return 4 * b * h * n * d * elem_bytes
