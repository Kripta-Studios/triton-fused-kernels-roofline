"""Timing + environment capture shared by the benchmark scripts."""

from __future__ import annotations

import platform
import subprocess
from typing import Callable

import torch
import triton
from triton.testing import do_bench


def time_ms(fn: Callable[[], object], warmup: int = 25, rep: int = 100) -> float:
    """Median runtime in ms. `do_bench` zeroes a 256 MB buffer between reps, so
    the 36 MB L2 is flushed and we measure DRAM-cold performance (otherwise a
    small RMSNorm would be timed against L2, not DRAM)."""
    return float(do_bench(fn, warmup=warmup, rep=rep, return_mode="median"))


def env_info() -> dict[str, str]:
    p = torch.cuda.get_device_properties(0)
    try:
        q = "--query-gpu=driver_version,clocks.max.sm,clocks.max.mem,power.limit"
        smi = subprocess.run(["nvidia-smi", q, "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        smi = "n/a"
    return {
        "gpu": p.name,
        "sm_count": str(p.multi_processor_count),
        "capability": f"sm_{p.major}{p.minor}",
        "vram_gb": f"{p.total_memory / 2**30:.1f}",
        "nvidia_smi(driver,max_sm_mhz,max_mem_mhz,power_limit)": smi,
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "triton": triton.__version__,
        "python": platform.python_version(),
        "os": platform.platform(),
    }


class ClockMonitor:
    """Samples SM/memory clock, temperature and power with nvidia-smi in a
    background thread. A laptop GPU's clock depends on temperature and power
    limit, so every result is reported together with the clocks it ran at."""

    FIELDS = ("clocks.sm", "clocks.mem", "temperature.gpu", "power.draw", "clocks_throttle_reasons.active")

    def __init__(self, period_s: float = 1.0) -> None:
        import threading

        self.period_s = period_s
        self.samples: list[list[str]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        cmd = ["nvidia-smi", f"--query-gpu={','.join(self.FIELDS)}", "--format=csv,noheader,nounits"]
        while not self._stop.is_set():
            try:
                line = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
                self.samples.append([f.strip() for f in line.split(",")])
            except (OSError, subprocess.SubprocessError):
                pass
            self._stop.wait(self.period_s)

    def __enter__(self) -> "ClockMonitor":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=15)

    def summary(self) -> dict[str, float | int | list[str]]:
        def col(i: int) -> list[float]:
            vals = []
            for s in self.samples:
                try:
                    vals.append(float(s[i]))
                except (ValueError, IndexError):
                    pass
            return sorted(vals)

        def med(v: list[float]) -> float:
            return v[len(v) // 2] if v else float("nan")

        sm, mem, temp, pw = col(0), col(1), col(2), col(3)
        return {
            "n_samples": len(self.samples),
            "sm_mhz_median": med(sm), "sm_mhz_min": sm[0] if sm else float("nan"),
            "sm_mhz_max": sm[-1] if sm else float("nan"),
            "mem_mhz_median": med(mem),
            "temp_c_median": med(temp), "temp_c_max": temp[-1] if temp else float("nan"),
            "power_w_median": med(pw),
            "throttle_reasons_seen": sorted({s[4] for s in self.samples if len(s) > 4}),
        }
