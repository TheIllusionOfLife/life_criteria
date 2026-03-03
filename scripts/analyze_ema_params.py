"""EMA memory parameter diagnostic — did evolution tune or ignore memory?

Since genome parameters are not exported in the current run summary, this
script uses observable proxies from the time-series data:

1. **Memory trace convergence**: Late-window variance of memory_mean.
   Low variance = EMA converged (evolution tuned gains > 0).
   High variance ≈ sham level = evolution drove gains toward 0.

2. **Memory trace level**: Late-window mean of memory_mean.
   If close to memory_target (0.5), EMA is actively correcting.
   If drifting toward IS means, gain is weak.

3. **Sham comparison**: Direct variance comparison between criterion8_on
   and sham conditions.  If indistinguishable, evolution "ignored" memory.

Reads
-----
    experiments/criterion8_criterion8_on.json
    experiments/criterion8_sham.json
    experiments/stress_famine_criterion8_on.json  (if available)
    experiments/stress_famine_sham.json            (if available)
    experiments/stress_boom_bust_criterion8_on.json (if available)
    experiments/stress_boom_bust_sham.json          (if available)

Writes
------
    experiments/ema_params_analysis.json

Usage
-----
    uv run python scripts/analyze_ema_params.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyses.results.statistics import (
    distribution_stats,
    run_paired_comparison,
)

_ROOT = Path(__file__).resolve().parent.parent
_EXP_DIR = _ROOT / "experiments"
_ANALYSIS_OUT = _EXP_DIR / "ema_params_analysis.json"

# Late window for convergence analysis
_LATE_START = 5_000
_LATE_END = 10_000


def _memory_late_stats(result: dict) -> dict:
    """Extract memory trace statistics in the late window [5k, 10k]."""
    samples = result.get("samples", [])
    late_mem = [
        s["memory_mean"]
        for s in samples
        if _LATE_START <= s["step"] <= _LATE_END and "memory_mean" in s
    ]
    late_mem_ch1 = [
        s.get("memory_mean_ch1", 0.0)
        for s in samples
        if _LATE_START <= s["step"] <= _LATE_END
    ]
    if not late_mem:
        return {"n_samples": 0}

    return {
        "n_samples": len(late_mem),
        "ch0": distribution_stats(np.array(late_mem)),
        "ch1": distribution_stats(np.array(late_mem_ch1)) if late_mem_ch1 else None,
        "variance": float(np.var(late_mem, ddof=1)) if len(late_mem) >= 2 else None,
        "variance_ch1": float(np.var(late_mem_ch1, ddof=1)) if len(late_mem_ch1) >= 2 else None,
    }


def _analyze_dataset(label: str, c8_path: Path, sham_path: Path) -> dict | None:
    """Compare memory dynamics between criterion8_on and sham for one dataset."""
    if not c8_path.exists() or not sham_path.exists():
        return None

    with open(c8_path) as f:
        c8_data = json.load(f)
    with open(sham_path) as f:
        sham_data = json.load(f)

    if len(c8_data) < 2 or len(sham_data) < 2:
        return None

    # Per-seed late-window variance
    c8_variances = []
    sham_variances = []
    c8_late_means = []
    sham_late_means = []

    for r in c8_data:
        stats = _memory_late_stats(r)
        if stats.get("variance") is not None:
            c8_variances.append(stats["variance"])
        if stats.get("ch0"):
            c8_late_means.append(stats["ch0"]["mean"])

    for r in sham_data:
        stats = _memory_late_stats(r)
        if stats.get("variance") is not None:
            sham_variances.append(stats["variance"])
        if stats.get("ch0"):
            sham_late_means.append(stats["ch0"]["mean"])

    result: dict = {
        "label": label,
        "n_c8": len(c8_data),
        "n_sham": len(sham_data),
        "c8_late_variance": distribution_stats(np.array(c8_variances)) if c8_variances else None,
        "sham_late_variance": (
            distribution_stats(np.array(sham_variances)) if sham_variances else None
        ),
        "c8_late_mean": (
            distribution_stats(np.array(c8_late_means)) if c8_late_means else None
        ),
        "sham_late_mean": (
            distribution_stats(np.array(sham_late_means)) if sham_late_means else None
        ),
    }

    # Paired comparison of late variances (c8 vs sham)
    n_paired = min(len(c8_variances), len(sham_variances))
    if n_paired >= 5:
        paired = run_paired_comparison(
            np.array(c8_variances[:n_paired]),
            np.array(sham_variances[:n_paired]),
        )
        result["variance_comparison"] = paired
        # Interpretation
        c8_med = np.median(c8_variances)
        sham_med = np.median(sham_variances)
        if c8_med < sham_med * 0.1:
            result["interpretation"] = (
                "EMA variance << sham variance: evolution tuned memory gains > 0 "
                "(memory is active and converging)"
            )
        elif c8_med < sham_med * 0.5:
            result["interpretation"] = (
                "EMA variance < sham variance: memory partially active"
            )
        else:
            result["interpretation"] = (
                "EMA variance ≈ sham variance: evolution may have driven gains "
                "toward 0 (memory ignored)"
            )

    return result


def run_analysis() -> dict:
    datasets = [
        ("normal_10k", _EXP_DIR / "criterion8_criterion8_on.json",
         _EXP_DIR / "criterion8_sham.json"),
        ("famine", _EXP_DIR / "stress_famine_criterion8_on.json",
         _EXP_DIR / "stress_famine_sham.json"),
        ("boom_bust", _EXP_DIR / "stress_boom_bust_criterion8_on.json",
         _EXP_DIR / "stress_boom_bust_sham.json"),
    ]

    analysis: dict = {"datasets": {}}
    for label, c8_path, sham_path in datasets:
        print(f"\nAnalysing EMA dynamics: {label}")
        result = _analyze_dataset(label, c8_path, sham_path)
        if result is None:
            print(f"  SKIP: data not available for {label}")
            continue
        analysis["datasets"][label] = result
        # Print summary
        c8v = result.get("c8_late_variance", {})
        sv = result.get("sham_late_variance", {})
        if c8v and sv:
            print(f"  c8_on  late var: median={c8v.get('median', 0):.6f}")
            print(f"  sham   late var: median={sv.get('median', 0):.6f}")
            ratio = c8v.get("median", 0) / max(sv.get("median", 1e-10), 1e-10)
            print(f"  ratio (c8/sham): {ratio:.4f}")
        interp = result.get("interpretation", "n/a")
        print(f"  interpretation: {interp}")

    _EXP_DIR.mkdir(exist_ok=True)
    with open(_ANALYSIS_OUT, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved: {_ANALYSIS_OUT}")
    return analysis


if __name__ == "__main__":
    run_analysis()
