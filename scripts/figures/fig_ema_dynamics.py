"""Figure: EMA memory convergence dynamics.

Two-panel figure:
  A  Memory trace time series — criterion8_on vs sham (median + IQR across seeds)
  B  Late-window variance comparison (boxplot: c8 vs sham across regimes)
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from figures._shared import FIG_DIR, PROJECT_ROOT

_EXP_DIR = PROJECT_ROOT / "experiments"

_DATASETS = [
    ("criterion8_criterion8_on.json", "criterion8_sham.json", "Normal"),
    ("stress_famine_criterion8_on.json", "stress_famine_sham.json", "Famine"),
    ("stress_boom_bust_criterion8_on.json", "stress_boom_bust_sham.json", "Boom-Bust"),
]


def _extract_memory_traces(results):
    """Extract memory_mean time series: dict of step→[values across seeds]."""
    traces = {}
    for r in results:
        for s in r.get("samples", []):
            step = s["step"]
            if "memory_mean" in s:
                traces.setdefault(step, []).append(s["memory_mean"])
    return traces


def _trace_summary(traces):
    """Convert traces to (steps, medians, q25, q75)."""
    steps = sorted(traces.keys())
    medians = [np.median(traces[s]) for s in steps]
    q25 = [np.percentile(traces[s], 25) for s in steps]
    q75 = [np.percentile(traces[s], 75) for s in steps]
    return steps, medians, q25, q75


def generate_ema_dynamics():
    """Generate EMA dynamics figure."""
    # Check if at least one dataset exists
    c8_path = _EXP_DIR / _DATASETS[0][0]
    if not c8_path.exists():
        print("  SKIP: criterion8_criterion8_on.json not found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    # Panel A: Time series for normal condition
    with open(c8_path) as f:
        c8_data = json.load(f)
    sham_path = _EXP_DIR / _DATASETS[0][1]
    if sham_path.exists():
        with open(sham_path) as f:
            sham_data = json.load(f)
    else:
        sham_data = []

    ax = axes[0]
    c8_traces = _extract_memory_traces(c8_data)
    steps, med, q25, q75 = _trace_summary(c8_traces)
    ax.plot(steps, med, color="#0072B2", lw=1.2, label="+Memory (EMA)")
    ax.fill_between(steps, q25, q75, color="#0072B2", alpha=0.15)

    if sham_data:
        sham_traces = _extract_memory_traces(sham_data)
        steps_s, med_s, q25_s, q75_s = _trace_summary(sham_traces)
        ax.plot(steps_s, med_s, color="#CC79A7", lw=1.2, label="Sham (random)")
        ax.fill_between(steps_s, q25_s, q75_s, color="#CC79A7", alpha=0.15)

    ax.set_xlabel("Step")
    ax.set_ylabel("Memory trace (mean)")
    ax.set_title("(A) Memory Convergence", fontsize=9)
    ax.legend(fontsize=7)

    # Panel B: Late-window variance across regimes
    ax = axes[1]
    regime_labels = []
    c8_vars = []
    sham_vars = []

    for c8_file, sham_file, label in _DATASETS:
        c8_p = _EXP_DIR / c8_file
        sham_p = _EXP_DIR / sham_file
        if not c8_p.exists() or not sham_p.exists():
            continue
        with open(c8_p) as f:
            c8 = json.load(f)
        with open(sham_p) as f:
            sh = json.load(f)

        c8_v = []
        sh_v = []
        for r in c8:
            vals = [s["memory_mean"] for s in r.get("samples", [])
                    if 5000 <= s["step"] <= 10000 and "memory_mean" in s]
            if len(vals) >= 2:
                c8_v.append(np.var(vals, ddof=1))
        for r in sh:
            vals = [s["memory_mean"] for s in r.get("samples", [])
                    if 5000 <= s["step"] <= 10000 and "memory_mean" in s]
            if len(vals) >= 2:
                sh_v.append(np.var(vals, ddof=1))

        if c8_v and sh_v:
            regime_labels.append(label)
            c8_vars.append(c8_v)
            sham_vars.append(sh_v)

    if regime_labels:
        x = np.arange(len(regime_labels))
        width = 0.35
        bp1 = ax.boxplot(
            c8_vars, positions=x - width / 2, widths=width * 0.8,
            patch_artist=True, showfliers=False,
        )
        bp2 = ax.boxplot(
            sham_vars, positions=x + width / 2, widths=width * 0.8,
            patch_artist=True, showfliers=False,
        )
        for patch in bp1["boxes"]:
            patch.set_facecolor("#0072B2")
            patch.set_alpha(0.4)
        for patch in bp2["boxes"]:
            patch.set_facecolor("#CC79A7")
            patch.set_alpha(0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(regime_labels)
        ax.set_ylabel("Late-window variance")
        ax.set_title("(B) Memory Stability by Regime", fontsize=9)
        ax.legend(
            [bp1["boxes"][0], bp2["boxes"][0]],
            ["+Memory", "Sham"],
            fontsize=7,
        )
    else:
        ax.set_visible(False)

    fig.tight_layout()
    out = FIG_DIR / "fig_ema_dynamics.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")
