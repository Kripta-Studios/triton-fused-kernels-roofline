"""Measure the two roofs of this GPU: DRAM bandwidth (copy kernel) and tensor-core
FP16 throughput (cuBLAS GEMM). Writes results/peak.json and results/env.json.

Run (from the repo root, through the shared GPU lock on the dev machine):
    python benchmarks/bench_peak.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fusedkernels.bench import ClockMonitor, env_info, time_ms  # noqa: E402
from fusedkernels.roofline import RTX_5070_TI_LAPTOP  # noqa: E402


@triton.jit
def _copy_kernel(X, Y, n, BLOCK: tl.constexpr):
    # The simplest possible streaming kernel: every byte is read once and written
    # once, with wide coalesced accesses. Its bandwidth is the practical ceiling
    # any memory-bound kernel (like RMSNorm) can hope to reach.
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(Y + offs, tl.load(X + offs, mask=mask), mask=mask)


def triton_copy(x: torch.Tensor, y: torch.Tensor) -> None:
    n = x.numel()
    _copy_kernel[(triton.cdiv(n, 4096),)](x, y, n, BLOCK=4096, num_warps=8)


def main() -> None:
    with ClockMonitor() as mon:
        out = run()
    out["clocks"] = mon.summary()
    print("clocks:", out["clocks"])
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "peak.json").write_text(json.dumps(out, indent=2))
    (ROOT / "results" / "env.json").write_text(json.dumps(env_info(), indent=2))


def run() -> dict[str, object]:
    out: dict[str, object] = {"copy": [], "gemm": []}
    # Sizes span "fits in the 40 MB L2" to "far bigger than L2". Max 512 MB per
    # buffer => ~1 GB VRAM, well inside the 6 GB budget.
    for mb in (8, 32, 128, 512):
        n = mb * 2**20 // 2
        x = torch.randn(n, device="cuda", dtype=torch.float16)
        y = torch.empty_like(x)
        bytes_moved = 2 * x.numel() * x.element_size()  # read + write
        for impl, fn in (("triton_copy", lambda: triton_copy(x, y)), ("torch_copy_", lambda: y.copy_(x))):
            ms = time_ms(fn)
            out["copy"].append({"impl": impl, "mb": mb, "ms": ms, "gbps": bytes_moved / ms / 1e6})
            print(f"copy {impl:12s} {mb:4d} MB  {bytes_moved / ms / 1e6:7.1f} GB/s", flush=True)
        assert torch.equal(x, y)
        del x, y
        torch.cuda.empty_cache()

    for dtype in (torch.float16, torch.bfloat16):
        for s in (2048, 4096, 8192):
            a = torch.randn(s, s, device="cuda", dtype=dtype)
            b = torch.randn(s, s, device="cuda", dtype=dtype)
            ms = time_ms(lambda: a @ b, rep=200)
            tflops = 2 * s**3 / ms / 1e9
            out["gemm"].append({"dtype": str(dtype), "size": s, "ms": ms, "tflops": tflops})
            print(f"gemm {str(dtype):15s} {s}^3  {tflops:6.1f} TFLOPS", flush=True)
            del a, b
    torch.cuda.empty_cache()

    best_bw = max(r["gbps"] for r in out["copy"] if r["mb"] >= 128)
    best_tf = max(r["tflops"] for r in out["gemm"])
    spec = RTX_5070_TI_LAPTOP
    out["summary"] = {
        "measured_dram_gbps": best_bw,
        "measured_tensor_tflops": best_tf,
        "datasheet_dram_gbps": spec.dram_gbps,
        "datasheet_tensor_fp16_fp32acc_tflops": spec.tensor_fp16_tflops,
        "bw_fraction_of_datasheet": best_bw / spec.dram_gbps,
        "tflops_fraction_of_datasheet": best_tf / spec.tensor_fp16_tflops,
    }
    print(json.dumps(out["summary"], indent=2))
    return out


if __name__ == "__main__":
    main()
