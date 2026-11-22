"""Helpers shared by the benchmark scripts (not part of the library)."""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable

# Windows quirk: Inductor's default cache dir embeds the user name, and with a
# non-ASCII user name Triton fails to decode the TRITON_CACHE_DIR env var that
# Inductor sets (UnicodeDecodeError). The temp dir itself is an ASCII 8.3 path,
# so put the cache there. Must happen before torch is imported.
if os.name == "nt" and "TORCHINDUCTOR_CACHE_DIR" not in os.environ:
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(tempfile.gettempdir(), "torchinductor_cache")

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))


def make_compiled(fn: Callable) -> Callable:
    """torch.compile with static shapes. We bench many shapes, so raise the
    recompile limit -- otherwise Dynamo silently falls back to eager after 8."""
    import torch._dynamo

    torch._dynamo.config.cache_size_limit = 128
    if hasattr(torch._dynamo.config, "recompile_limit"):
        torch._dynamo.config.recompile_limit = 128
    return torch.compile(fn, dynamic=False)


def write_csv(rows: list[dict], name: str) -> None:
    RESULTS.mkdir(exist_ok=True)
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(RESULTS / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def write_json(obj: object, name: str) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def free() -> None:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
