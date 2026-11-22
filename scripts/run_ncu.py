"""Run Nsight Compute on scripts/ncu_target.py and save a small CSV of key metrics.

    python scripts/run_ncu.py [--ncu PATH_TO_NCU]

Writes results/ncu_metrics.csv on success, or results/ncu_attempt.txt with
ncu's exact output on failure (e.g. an ncu build too old for this GPU, or
ERR_NVGPUCTRPERM when GPU performance counters are restricted to admins).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NCU = r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.1\target\windows-desktop-win7-x64\ncu.exe"
METRICS = [
    "gpu__time_duration.sum",
    "dram__bytes.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",  # = achieved occupancy
]


def redact(text: str) -> str:
    """Keep local user paths out of committed result files."""
    home = str(Path.home())
    return text.replace(home, "~").replace(home.replace("\\", "/"), "~")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncu", default=shutil.which("ncu") or DEFAULT_NCU)
    args = ap.parse_args()
    cmd = [
        args.ncu, "--profile-from-start", "off", "--target-processes", "all",
        "--metrics", ",".join(METRICS), "--csv",
        sys.executable, str(ROOT / "scripts" / "ncu_target.py"),
    ]
    print("running:", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=600)
    out = redact(p.stdout + "\n" + p.stderr)
    csv_lines = [ln for ln in redact(p.stdout).splitlines() if ln.startswith('"')]
    (ROOT / "results").mkdir(exist_ok=True)
    if p.returncode == 0 and len(csv_lines) > 1:
        (ROOT / "results" / "ncu_metrics.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
        print("\n".join(csv_lines))
    else:
        (ROOT / "results" / "ncu_attempt.txt").write_text(
            f"command: {redact(' '.join(cmd))}\nreturn code: {p.returncode}\n\n{out[-6000:]}", encoding="utf-8")
        print(out[-6000:])
        print(f"ncu failed (rc={p.returncode}); see results/ncu_attempt.txt")


if __name__ == "__main__":
    main()
