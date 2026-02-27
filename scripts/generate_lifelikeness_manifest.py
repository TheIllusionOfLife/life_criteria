"""Generate experiments/lifelikeness_manifest.json from lifelikeness_analysis.json.

Compact manifest committed to the repo (full per-seed JSONs are gitignored).
Run AFTER analyze_lifelikeness.py has written lifelikeness_analysis.json.

Usage
-----
    uv run python scripts/generate_lifelikeness_manifest.py
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_EXP_DIR = _ROOT / "experiments"
_ANALYSIS_PATH = _EXP_DIR / "lifelikeness_analysis.json"
_MANIFEST_PATH = _EXP_DIR / "lifelikeness_manifest.json"

_CONDITIONS = ["normal_graph", "shift_graph", "normal_graph_memory", "shift_graph_memory"]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main() -> None:
    if not _ANALYSIS_PATH.exists():
        print(f"ERROR: {_ANALYSIS_PATH} not found — run analyze_lifelikeness.py first.")
        raise SystemExit(1)

    with open(_ANALYSIS_PATH) as f:
        analysis = json.load(f)

    # Collect seeds from per-condition data files + validate
    condition_seeds: dict[str, list[int]] = {}
    for cond in _CONDITIONS:
        path = _EXP_DIR / f"lifelikeness_t1_{cond}.json"
        if not path.exists():
            print(f"WARNING: {path} not found — condition '{cond}' will be absent from manifest.")
            continue
        with open(path) as f:
            data = json.load(f)
        condition_seeds[cond] = sorted(r["seed"] for r in data)

    if not condition_seeds:
        raise SystemExit("ERROR: No condition data files found.")

    # Use the first available condition as the reference seed set
    reference_cond = next(iter(condition_seeds))
    seeds_used = condition_seeds[reference_cond]

    # Cross-validate seeds across all available conditions
    for cond, seeds in condition_seeds.items():
        if seeds != seeds_used:
            raise SystemExit(
                f"ERROR: Seed mismatch — {cond} has {seeds}, {reference_cond} has {seeds_used}."
            )

    # Validate held-out seed policy (100–199)
    if not seeds_used:
        raise SystemExit("ERROR: No seeds found. Cannot generate manifest without seed data.")
    bad_seeds = [s for s in seeds_used if not (100 <= s <= 199)]
    if bad_seeds:
        raise SystemExit(
            f"ERROR: Seeds outside held-out range 100–199: {bad_seeds}. "
            "Calibration seeds 0–99 must not appear in published results."
        )
    if len(seeds_used) < 30:
        print(f"WARNING: Only {len(seeds_used)} seeds found (expected 30). Partial run?")

    # Extract summary statistics
    per_cond = analysis.get("per_condition_summary", {})
    decision = analysis.get("decision", {})
    memory_comparisons = analysis.get("memory_comparisons", {})

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": "lifelikeness",
        "steps": 10_000,
        "sample_every": 100,
        "seeds": seeds_used,
        "n_seeds": len(seeds_used),
        "conditions": list(condition_seeds.keys()),
        "condition_overrides": {
            "normal_graph": {
                "metabolism_mode": "graph",
                "max_alive_organisms": 100,
            },
            "shift_graph": {
                "metabolism_mode": "graph",
                "max_alive_organisms": 100,
                "environment_shift_step": 5000,
                "environment_shift_resource_rate": 0.003,
            },
            "normal_graph_memory": {
                "metabolism_mode": "graph",
                "max_alive_organisms": 100,
                "enable_memory": True,
            },
            "shift_graph_memory": {
                "metabolism_mode": "graph",
                "max_alive_organisms": 100,
                "enable_memory": True,
                "environment_shift_step": 5000,
                "environment_shift_resource_rate": 0.003,
            },
        },
        "statistics": {
            "per_condition": per_cond,
            "decision": decision,
            "memory_comparisons": memory_comparisons,
        },
        "analysis_digest": _file_digest(_ANALYSIS_PATH),
        "script_name": "experiment_lifelikeness.py",
        "generator_script": "generate_lifelikeness_manifest.py",
        "git_commit": _git_commit(),
    }

    with open(_MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"Saved: {_MANIFEST_PATH}")

    # Human-readable summary
    seed_lo = seeds_used[0]
    seed_hi = seeds_used[-1]
    print("\nResults summary:")
    print(f"  Seeds: {seed_lo}–{seed_hi} (n={len(seeds_used)})")
    print(f"  Decision: {decision.get('criterion8_candidate', 'n/a')}")
    print(f"  Rule: {decision.get('triggered_rule', 'n/a')}")

    print(f"\n  {'Condition':<25}  {'Median AUC':>12}  {'Ext@10k':>8}  {'Div slope':>10}")
    print(f"  {'-' * 25}  {'-' * 12}  {'-' * 8}  {'-' * 10}")
    for cond in condition_seeds:
        cs = per_cond.get(cond, {})
        med = cs.get("survival_auc_median")
        med_s = f"{med:.1f}" if med is not None else "n/a"
        ext = cs.get("extinction_fraction_10k")
        ext_s = f"{ext:.3f}" if ext is not None else "n/a"
        div = cs.get("diversity_slope_median")
        div_s = f"{div:.4f}" if div is not None else "n/a"
        print(f"  {cond:<25}  {med_s:>12}  {ext_s:>8}  {div_s:>10}")

    if memory_comparisons:
        print("\n  Memory effect (pairwise):")
        for label, comp in memory_comparisons.items():
            auc = comp.get("survival_auc", {})
            p_adj = auc.get("mwu_p_adj")
            d = auc.get("cohen_d")
            p_str = f"{p_adj:.4f}" if p_adj is not None else "n/a"
            d_str = f"{d:.3f}" if d is not None else "n/a"
            sig = auc.get("significant_adj005")
            sig_str = "sig" if sig else "n.s." if sig is not None else "?"
            print(f"    {label}: p_adj={p_str}  d={d_str}  ({sig_str})")


if __name__ == "__main__":
    main()
