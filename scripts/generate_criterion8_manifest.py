"""Generate experiments/criterion8_manifest.json from criterion8_analysis.json.

Compact manifest committed to the repo (full per-seed JSONs are gitignored).
Run AFTER analyze_criterion8.py has written criterion8_analysis.json.

Usage
-----
    uv run python scripts/generate_criterion8_manifest.py
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_EXP_DIR = _ROOT / "experiments"
_ANALYSIS_PATH = _EXP_DIR / "criterion8_analysis.json"
_MANIFEST_PATH = _EXP_DIR / "criterion8_manifest.json"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main() -> None:
    if not _ANALYSIS_PATH.exists():
        print(f"ERROR: {_ANALYSIS_PATH} not found — run analyze_criterion8.py first.")
        raise SystemExit(1)

    with open(_ANALYSIS_PATH) as f:
        analysis = json.load(f)

    summaries = analysis.get("summaries", {})
    pairwise = analysis.get("pairwise_vs_baseline", {})
    memory_stability = analysis.get("memory_stability", {})

    # Condition stats table
    conditions_stats: dict[str, dict] = {}
    for cond in ["baseline", "criterion8_on", "criterion8_ablated", "sham"]:
        s = summaries.get(cond, {}).get("survival_auc", {})
        pw = pairwise.get(cond, {})
        conditions_stats[cond] = {
            "n_seeds": summaries.get(cond, {}).get("n_seeds", 0),
            "median_auc": s.get("median"),
            "mean_auc": s.get("mean"),
            "std_auc": s.get("std"),
            "vs_baseline_mwu_p_adj": pw.get("vs_baseline_mwu_p_adj"),
            "vs_baseline_cohen_d": pw.get("vs_baseline_cohen_d"),
            "vs_baseline_significant_adj005": pw.get("vs_baseline_significant_adj005"),
        }

    # Seeds used (extracted from data sources, or from first condition's n_seeds)
    baseline_results_path = _EXP_DIR / "criterion8_baseline.json"
    seeds_used: list[int] = []
    if baseline_results_path.exists():
        with open(baseline_results_path) as f:
            results = json.load(f)
        seeds_used = sorted(r["seed"] for r in results)

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": "criterion8",
        "steps": 10_000,
        "sample_every": 100,
        "seeds": seeds_used,
        "n_seeds": len(seeds_used),
        "conditions": ["baseline", "criterion8_on", "criterion8_ablated", "sham"],
        "condition_overrides": {
            "baseline": {"max_alive_organisms": 100, "enable_memory": False},
            "criterion8_on": {"max_alive_organisms": 100, "enable_memory": True},
            "criterion8_ablated": {
                "max_alive_organisms": 100,
                "enable_memory": True,
                "ablation_targets": ["memory"],
                "ablation_step": 5000,
            },
            "sham": {
                "max_alive_organisms": 100,
                "enable_memory": True,
                "enable_sham_process": True,
            },
        },
        "statistics": {
            "conditions": conditions_stats,
            "memory_stability": {
                "criterion8_on_mean_late_variance": memory_stability.get(
                    "criterion8_on_mean_late_variance"
                ),
                "sham_mean_late_variance": memory_stability.get("sham_mean_late_variance"),
                "mwu_c8_vs_sham_p": memory_stability.get("mwu_c8_vs_sham_p"),
                "interpretation": memory_stability.get("interpretation"),
            },
            "orthogonality": analysis.get("orthogonality", {}),
            "primary_outcome_met": (
                pairwise.get("criterion8_on", {}).get("vs_baseline_significant_adj005")
                is True
            ),
        },
        "analysis_digest": _file_digest(_ANALYSIS_PATH),
        "script_name": "experiment_criterion8.py",
        "git_commit": _git_commit(),
    }

    with open(_MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved: {_MANIFEST_PATH}")

    # Human-readable summary
    print("\nResults summary:")
    print(f"  Seeds: {seeds_used[0] if seeds_used else '?'}–{seeds_used[-1] if seeds_used else '?'} (n={len(seeds_used)})")
    print(f"  Primary outcome met (criterion8_on p_adj<0.05): {manifest['statistics']['primary_outcome_met']}")
    print()
    print(f"  {'Condition':<25}  {'Median AUC':>12}  {'p_adj':>8}  {'Cohen d':>8}")
    print(f"  {'-'*25}  {'-'*12}  {'-'*8}  {'-'*8}")
    for cond in ["baseline", "criterion8_on", "criterion8_ablated", "sham"]:
        cs = conditions_stats[cond]
        med = f"{cs['median_auc']:.1f}" if cs["median_auc"] is not None else "n/a"
        p = f"{cs['vs_baseline_mwu_p_adj']:.4f}" if cs["vs_baseline_mwu_p_adj"] is not None else "—"
        d = f"{cs['vs_baseline_cohen_d']:.3f}" if cs["vs_baseline_cohen_d"] is not None else "—"
        print(f"  {cond:<25}  {med:>12}  {p:>8}  {d:>8}")
    ms = manifest["statistics"]["memory_stability"]
    mem_p = ms["mwu_c8_vs_sham_p"]
    print(f"\n  Memory stability (c8_on vs sham): p={mem_p:.4f}" if mem_p is not None else "\n  Memory stability: n/a")


if __name__ == "__main__":
    main()
