"""Roofline model: attainable = min(peak_compute, peak_bandwidth * arithmetic_intensity).

(Williams, Waterman & Patterson, CACM 2009.)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GpuSpec:
    name: str
    dram_gbps: float          # GB/s (1e9 bytes/s)
    tensor_fp16_tflops: float  # dense FP16 in / FP32 accumulate tensor-core peak
    fp32_tflops: float         # non-tensor FP32 (CUDA cores) peak

    @property
    def ridge_point(self) -> float:
        """FLOP/byte where the two roofs meet: kernels left of it are memory bound."""
        return self.tensor_fp16_tflops * 1e12 / (self.dram_gbps * 1e9)


# Datasheet numbers for the GPU in this machine (laptop part, NOT the 16 GB
# desktop RTX 5070 Ti, which has a 256-bit bus and 896 GB/s). Sources in docs/SOTA.md:
# 46 SMs, 192-bit GDDR7 @ 28 Gbps -> 192/8 * 28 = 672 GB/s; boost ~1.82 GHz.
# FP16 tensor w/ FP32 accumulate ~55 TFLOPS dense (GeForce parts run FP32-accumulate
# at half the FP16-accumulate rate of ~110 TFLOPS). The real clock (and thus the
# peak) depends on the laptop's power limit, which is why we also *measure* both roofs.
RTX_5070_TI_LAPTOP = GpuSpec(
    name="RTX 5070 Ti Laptop (datasheet)",
    dram_gbps=672.0,
    tensor_fp16_tflops=54.95,
    fp32_tflops=21.43,
)


def attainable_tflops(ai: float, peak_tflops: float, peak_gbps: float) -> float:
    return min(peak_tflops, peak_gbps * 1e9 * ai / 1e12)


def achieved(flops: float, bytes_moved: float, ms: float) -> dict[str, float]:
    """Convert a measured runtime into roofline coordinates."""
    s = ms * 1e-3
    return {
        "ms": ms,
        "tflops": flops / s / 1e12,
        "gbps": bytes_moved / s / 1e9,
        "ai": flops / bytes_moved,
    }
