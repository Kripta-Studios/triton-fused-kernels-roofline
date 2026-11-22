"""Numerics of the fused RMSNorm vs an fp32 PyTorch oracle."""

import pytest
import torch

from fusedkernels import rmsnorm
from fusedkernels.reference import add_rmsnorm_ref, rmsnorm_native, rmsnorm_ref

# One rounding of the output to fp16/bf16 dominates the error; these are ~2 ulp.
TOL = {torch.float16: dict(atol=2e-3, rtol=2e-3), torch.bfloat16: dict(atol=1.6e-2, rtol=1.6e-2)}
SHAPES = [(1, 1024), (7, 1000), (64, 4096), (3, 5, 2048), (16, 8192), (4, 16384)]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", SHAPES)
def test_rmsnorm_matches_reference(shape, dtype):
    x = torch.randn(shape, device="cuda", dtype=dtype) * 3
    w = torch.randn(shape[-1], device="cuda", dtype=dtype)
    torch.testing.assert_close(rmsnorm(x, w), rmsnorm_ref(x, w), **TOL[dtype])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", SHAPES)
def test_fused_residual_add(shape, dtype):
    x = torch.randn(shape, device="cuda", dtype=dtype)
    r = torch.randn(shape, device="cuda", dtype=dtype)
    w = torch.randn(shape[-1], device="cuda", dtype=dtype)
    y, h = rmsnorm(x, w, residual=r)
    y_ref, h_ref = add_rmsnorm_ref(x, r, w)
    # The residual stream must be *bit-identical* to the unfused x + r.
    assert torch.equal(h, h_ref)
    torch.testing.assert_close(y, y_ref, **TOL[dtype])


def test_matches_torch_native_and_large_values():
    # fp16 values of 200 would overflow an fp16 sum of squares (200^2*4096 >> 65504);
    # fp32 accumulation in the kernel must handle it.
    x = torch.full((8, 4096), 200.0, device="cuda", dtype=torch.float16)
    w = torch.ones(4096, device="cuda", dtype=torch.float16)
    y = rmsnorm(x, w)
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y, rmsnorm_native(x, w), **TOL[torch.float16])


def test_non_contiguous_rows():
    base = torch.randn(32, 2 * 1024, device="cuda", dtype=torch.float16)
    x = base[:, :1024]  # row stride 2048, inner stride 1
    w = torch.randn(1024, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(rmsnorm(x, w), rmsnorm_ref(x, w), **TOL[torch.float16])
