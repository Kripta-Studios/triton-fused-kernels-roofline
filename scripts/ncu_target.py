"""Tiny, deterministic workload for Nsight Compute.

Warm-up (JIT compile + autotune) runs *outside* the profiled region; then each
kernel is launched exactly once between cudaProfilerStart/Stop, so ncu run with
`--profile-from-start off` captures one clean launch per kernel:

    ncu --profile-from-start off --target-processes all \
        --metrics gpu__time_duration.sum,dram__bytes.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active \
        --csv python scripts/ncu_target.py

(see scripts/run_ncu.py, which wraps exactly this command.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fusedkernels import causal_attention, rmsnorm  # noqa: E402

DT = torch.bfloat16


def main() -> None:
    x = torch.randn(16384, 4096, device="cuda", dtype=DT)  # 128 MB, far beyond L2
    r = torch.randn_like(x)
    w = torch.randn(4096, device="cuda", dtype=DT)
    q, k, v = (torch.randn(1, 16, 4096, 128, device="cuda", dtype=DT) for _ in range(3))

    for _ in range(3):  # compile + autotune, not profiled
        rmsnorm(x, w)
        rmsnorm(x, w, residual=r)
        causal_attention(q, k, v)
    torch.cuda.synchronize()

    torch.cuda.profiler.start()
    rmsnorm(x, w)
    rmsnorm(x, w, residual=r)
    causal_attention(q, k, v)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()


if __name__ == "__main__":
    main()
