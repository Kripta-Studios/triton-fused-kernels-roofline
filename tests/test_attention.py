"""Numerics of the causal FlashAttention-2 kernel vs fp32 math and PyTorch SDPA."""

import pytest
import torch

from fusedkernels import attention_flops, causal_attention
from fusedkernels.reference import attention_ref, attention_sdpa

TOL = {torch.float16: dict(atol=2e-3, rtol=2e-3), torch.bfloat16: dict(atol=1.6e-2, rtol=1.6e-2)}


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "b,h,n,d",
    [
        (1, 1, 128, 64),
        (2, 3, 256, 64),
        (1, 2, 1000, 128),  # N not a multiple of any block size
        (1, 4, 77, 32),     # shorter than one query tile
        (2, 2, 513, 128),
        (1, 1, 1, 64),      # single token
    ],
)
def test_attention_matches_fp32_reference(b, h, n, d, dtype):
    q, k, v = (torch.randn(b, h, n, d, device="cuda", dtype=dtype) for _ in range(3))
    out = causal_attention(q, k, v)
    ref = attention_ref(q, k, v)
    torch.testing.assert_close(out, ref, **TOL[dtype])
    # And it should be at least as close to fp32 truth as PyTorch's own kernel (x2 slack).
    err = (out.float() - ref.float()).abs().max()
    err_sdpa = (attention_sdpa(q, k, v).float() - ref.float()).abs().max()
    assert err <= 2 * err_sdpa + 1e-3


def test_online_softmax_is_stable_for_large_logits():
    # q, k ~ N(0, 64), D = 64: scaled logits have std ~64 and reach ~+-300, far past
    # the ~88 where exp() overflows fp32 without max subtraction. The running
    # max in the online softmax must keep everything finite.
    q = torch.randn(1, 2, 256, 64, device="cuda", dtype=torch.float16) * 8
    k = torch.randn(1, 2, 256, 64, device="cuda", dtype=torch.float16) * 8
    v = torch.randn(1, 2, 256, 64, device="cuda", dtype=torch.float16)
    out = causal_attention(q, k, v)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, attention_ref(q, k, v), atol=5e-3, rtol=5e-3)


def test_causality():
    # Changing future tokens must not change earlier outputs.
    q, k, v = (torch.randn(1, 2, 300, 64, device="cuda", dtype=torch.float16) for _ in range(3))
    out1 = causal_attention(q, k, v)
    k2, v2 = k.clone(), v.clone()
    k2[:, :, 150:] = torch.randn_like(k2[:, :, 150:])
    v2[:, :, 150:] = torch.randn_like(v2[:, :, 150:])
    out2 = causal_attention(q, k2, v2)
    assert torch.equal(out1[:, :, :150], out2[:, :, :150])


def test_non_contiguous_layout():
    # [B, N, H, D] storage viewed as [B, H, N, D] (the layout after a QKV projection).
    q, k, v = (torch.randn(2, 200, 4, 64, device="cuda", dtype=torch.float16).transpose(1, 2)
               for _ in range(3))
    torch.testing.assert_close(causal_attention(q, k, v), attention_ref(q, k, v), atol=2e-3, rtol=2e-3)


def test_flop_count():
    assert attention_flops(1, 1, 2, 4) == 4 * 3 * 4  # 3 (q,k) pairs, 4 FLOP per pair per dim
