"""Directed coupling analysis for criterion interactions.

Combines time-lagged correlation, Granger-style F-tests, and transfer entropy
to measure directed information flow between physiology criteria.

Usage:
    uv run python scripts/analyze_coupling.py
    uv run python scripts/analyze_coupling.py --robustness-profile fast

Implementation lives in ``analyses/coupling/``; this file is a thin dispatcher.
"""

from __future__ import annotations

import argparse

from analyses.coupling import ROBUSTNESS_PROFILES, main

# Re-export symbols used by existing tests and scripts
from analyses.coupling.transfer_entropy import (  # noqa: F401
    phase_randomize,
    te_robustness_summary,
    transfer_entropy_lag1,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robustness-profile",
        choices=sorted(ROBUSTNESS_PROFILES.keys()),
        default="full",
        help="Runtime/precision profile for TE robustness computations.",
    )
    args = parser.parse_args()
    main(robustness_profile=args.robustness_profile)
