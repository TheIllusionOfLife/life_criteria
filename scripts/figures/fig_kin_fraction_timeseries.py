"""Figure: kin_fraction time series showing signal degeneration under pop cap.

Two panels:
  A  kin_fraction_mean over time for Candidate B conditions (famine regime)
  B  Comparison: cap=100 vs cap=400 kin_fraction trajectory
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from figures._shared import FIG_DIR, PROJECT_ROOT

_EXP_DIR = PROJECT_ROOT / "experiments"

_COLORS = {
    "baseline": "#000000",
    "candidateB_on": "#0072B2",
    "candidateB_ablated": "#D55E00",
    "sham": "#CC79A7",
}
_LABELS = {
    "baseline": "Baseline",
    "candidateB_on": "+Kin-Sensing",
    "candidateB_ablated": "+Kin-Sensing (ablated)",
    "sham": "Sham",
}


def _extract_kf_traces(results):
    """Extract kin_fraction_mean traces: step→[values]."""
    traces = {}
    for r in results:
        for s in r.get("samples", []):
            step = s["step"]
            kf = s.get("kin_fraction_mean", 0.0)
            traces.setdefault(step, []).append(kf)
    return traces


def _trace_summary(traces):
    steps = sorted(traces.keys())
    medians = [np.median(traces[s]) for s in steps]
    q25 = [np.percentile(traces[s], 25) for s in steps]
    q75 = [np.percentile(traces[s], 75) for s in steps]
    return steps, medians, q25, q75


def generate_kin_fraction_timeseries():
    """Generate kin_fraction time series figure."""
    # Panel A: kin_fraction for all conditions under famine (cap=100)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    ax = axes[0]
    has_data = False
    for cond in ["baseline", "candidateB_on", "candidateB_ablated", "sham"]:
        path = _EXP_DIR / f"candidateB_famine_{cond}.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        traces = _extract_kf_traces(data)
        if not traces:
            continue
        steps, med, q25, q75 = _trace_summary(traces)
        color = _COLORS.get(cond, "#999")
        ax.plot(steps, med, color=color, lw=1.2, label=_LABELS.get(cond, cond))
        ax.fill_between(steps, q25, q75, color=color, alpha=0.1)
        has_data = True

    if has_data:
        ax.axvline(3000, color="red", ls="--", lw=0.8, alpha=0.5, label="Famine onset")
        ax.set_xlabel("Step")
        ax.set_ylabel("Kin fraction (mean)")
        ax.set_title("(A) Kin Signal — Cap=100", fontsize=9)
        ax.legend(fontsize=6, loc="upper right")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(A) Kin Signal — Cap=100", fontsize=9)

    # Panel B: cap=100 vs cap=400 for candidateB_on (famine)
    ax = axes[1]
    has_b = False
    for prefix, label, color in [
        ("candidateB_famine_candidateB_on", "Cap=100", "#0072B2"),
        ("relaxed_cap_famine_candidateB_on", "Cap=400", "#E69F00"),
    ]:
        path = _EXP_DIR / f"{prefix}.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        traces = _extract_kf_traces(data)
        if not traces:
            continue
        steps, med, q25, q75 = _trace_summary(traces)
        ax.plot(steps, med, color=color, lw=1.2, label=label)
        ax.fill_between(steps, q25, q75, color=color, alpha=0.1)
        has_b = True

    if has_b:
        ax.set_xlabel("Step")
        ax.set_ylabel("Kin fraction (mean)")
        ax.set_title("(B) Cap=100 vs Cap=400", fontsize=9)
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(B) Cap=100 vs Cap=400", fontsize=9)

    fig.tight_layout()
    out = FIG_DIR / "fig_kin_fraction_timeseries.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved {out}")
