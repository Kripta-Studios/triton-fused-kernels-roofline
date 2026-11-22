"""Benchmark the causal FA2 Triton kernel vs PyTorch eager, torch.compile and
F.scaled_dot_product_attention (each SDPA backend separately). Writes
results/attention.csv.
"""

from __future__ import annotations

from _common import free, make_compiled, write_csv, write_json  # sets env before torch import
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from fusedkernels import attention_bytes, attention_flops, causal_attention
from fusedkernels.bench import ClockMonitor, time_ms
from fusedkernels.reference import attention_eager, attention_sdpa
from fusedkernels.roofline import achieved

DTYPE = torch.bfloat16
B, H = 1, 16                # Qwen3-0.6B has 16 query heads
SEQ = (512, 1024, 2048, 4096, 8192, 16384)
EAGER_MAX_N = 4096          # eager materializes B*H*N^2 scores: 16*4096^2*4 B = 1 GB in fp32
BACKENDS = {
    "sdpa_flash": SDPBackend.FLASH_ATTENTION,
    "sdpa_efficient": SDPBackend.EFFICIENT_ATTENTION,
    "sdpa_cudnn": SDPBackend.CUDNN_ATTENTION,
}


def sdpa_with(backend: SDPBackend):
    def fn(q, k, v):
        with sdpa_kernel(backend):
            return attention_sdpa(q, k, v)
    return fn


def main() -> None:
    compiled = make_compiled(attention_eager)
    rows, unavailable = [], {}
    with ClockMonitor() as mon:
        for d in (64, 128):
            for n in SEQ:
                q, k, v = (torch.randn(B, H, n, d, device="cuda", dtype=DTYPE) for _ in range(3))
                impls = {}
                if n <= EAGER_MAX_N:
                    impls["eager"] = lambda: attention_eager(q, k, v)
                    impls["torch_compile"] = lambda: compiled(q, k, v)
                for name, be in BACKENDS.items():
                    fn = sdpa_with(be)
                    try:
                        fn(q, k, v)
                        impls[name] = lambda fn=fn: fn(q, k, v)
                    except RuntimeError as e:  # backend not built / not supported on this GPU
                        unavailable[name] = str(e).splitlines()[0][:200]
                impls["triton_fa2"] = lambda: causal_attention(q, k, v)

                flops = attention_flops(B, H, n, d)
                nbytes = attention_bytes(B, H, n, d, q.element_size())
                for impl, fn in impls.items():
                    ms = time_ms(fn, rep=50)
                    a = achieved(flops, nbytes, ms)
                    rows.append({"B": B, "H": H, "N": n, "D": d, "dtype": "bf16", "impl": impl, **a})
                    print(f"D={d:3d} N={n:5d} {impl:15s} {ms:8.3f} ms {a['tflops']:6.1f} TFLOPS",
                          flush=True)
                del q, k, v, impls
                free()
    write_csv(rows, "attention.csv")
    write_json({"clocks": mon.summary(), "sdpa_unavailable": unavailable}, "attention_meta.json")
    print(mon.summary(), unavailable)


if __name__ == "__main__":
    main()
