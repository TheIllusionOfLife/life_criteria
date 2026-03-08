"""Compute genome gain statistics from criterion8_on experiment data.

Extracts final-timestep memory gain offsets (g_genome) from each seed,
then reports mean(|g|), std(g), and max(|g|) across all seeds and channels.

These values are cited in paper/main.tex (Section: Mechanism Verification).

Usage:
    uv run python scripts/compute_gain_stats.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

DATA_PATH = Path("experiments/criterion8_criterion8_on.json")


def main() -> None:
    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found. Run the genome re-run first:", file=sys.stderr)
        print(
            "  uv run python scripts/experiment_criterion8.py --tier 1 --conditions criterion8_on",
            file=sys.stderr,
        )
        sys.exit(1)

    data = json.loads(DATA_PATH.read_text())
    finals = [seed_run["samples"][-1] for seed_run in data]
    n_seeds = len(finals)

    g0 = [s["memory_gain_is0_mean"] for s in finals]
    g1 = [s["memory_gain_is1_mean"] for s in finals]
    all_g = g0 + g1
    abs_g = np.abs(all_g)

    print(f"Source: {DATA_PATH}")
    print(f"Seeds:  {n_seeds}")
    print(f"Values: {len(all_g)} (2 channels x {n_seeds} seeds)")
    print()
    print(f"mean(|g|) = {np.mean(abs_g):.4f}")
    print(f"std(g)    = {np.std(all_g):.4f}")
    print(f"max(|g|)  = {np.max(abs_g):.4f}")
    print()
    print(f"g0 range: [{min(g0):.4f}, {max(g0):.4f}]")
    print(f"g1 range: [{min(g1):.4f}, {max(g1):.4f}]")


if __name__ == "__main__":
    main()
