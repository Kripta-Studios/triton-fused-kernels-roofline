"""Benchmark fused RMSNorm (+ residual add) vs PyTorch eager, F.rms_norm and
torch.compile. Writes results/rmsnorm.csv.

GB/s is *effective* bandwidth: the fused kernel's minimum traffic divided by
each implementation's time, so all implementations are compared on the same
useful work (an unfused impl moves more bytes and therefore scores lower).
"""

from __future__ import annotations

from _common import free, make_compiled, write_csv, write_json  # sets env before torch import
import torch

from fusedkernels import rmsnorm, rmsnorm_bytes, rmsnorm_flops
from fusedkernels.bench import ClockMonitor, time_ms
from fusedkernels.reference import rmsnorm_native, rmsnorm_ref
from fusedkernels.roofline import achieved

DTYPE = torch.bfloat16
HIDDEN = (1024, 4096, 8192)          # Qwen3-0.6B, Llama-3-8B, Llama-3-70B
TOKENS = (1, 64, 1024, 4096, 16384)  # decode .. long prefill


def add_then_composite(x, r, w):
    h = x + r
    return rmsnorm_ref(h, w), h


def add_then_native(x, r, w):
    h = x + r
    return rmsnorm_native(h, w), h


def main() -> None:
    compiled_plain = make_compiled(rmsnorm_ref)
    compiled_res = make_compiled(add_then_composite)
    rows = []
    with ClockMonitor() as mon:
        for n in HIDDEN:
            w = torch.randn(n, device="cuda", dtype=DTYPE)
            for m in TOKENS:
                x = torch.randn(m, n, device="cuda", dtype=DTYPE)
                r = torch.randn(m, n, device="cuda", dtype=DTYPE)
                for residual in (False, True):
                    if residual:
                        impls = {
                            "eager": lambda: add_then_composite(x, r, w),
                            "torch_native": lambda: add_then_native(x, r, w),
                            "torch_compile": lambda: compiled_res(x, r, w),
                            "triton_fused": lambda: rmsnorm(x, w, residual=r),
                        }
                    else:
                        impls = {
                            "eager": lambda: rmsnorm_ref(x, w),
                            "torch_native": lambda: rmsnorm_native(x, w),
                            "torch_compile": lambda: compiled_plain(x, w),
                            "triton_fused": lambda: rmsnorm(x, w),
                        }
                    nbytes = rmsnorm_bytes(m, n, x.element_size(), residual)
                    flops = rmsnorm_flops(m, n, residual)
                    eager_ms = None
                    for impl, fn in impls.items():
                        ms = time_ms(fn)
                        eager_ms = eager_ms or ms
                        a = achieved(flops, nbytes, ms)
                        rows.append({"residual": residual, "tokens": m, "hidden": n,
                                     "dtype": "bf16", "impl": impl, **a,
                                     "speedup_vs_eager": eager_ms / ms})
                        print(f"res={residual!s:5} M={m:5d} N={n:4d} {impl:14s} "
                              f"{ms * 1e3:8.1f} us {a['gbps']:6.1f} GB/s", flush=True)
                del x, r
                free()
    write_csv(rows, "rmsnorm.csv")
    write_json(mon.summary(), "rmsnorm_clocks.json")
    print(mon.summary())


if __name__ == "__main__":
    main()
