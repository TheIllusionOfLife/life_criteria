"""Generate all paper figures. Delegates to scripts/figures/ subpackage.

Utility functions from the shared helper are re-exported here for backward
compatibility with tests and notebooks that import from this module.
"""

import sys
from pathlib import Path

# Ensure scripts/ is on the path when called directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from figures import (
    generate_criterion8,
    generate_ablation_distributions,
    generate_architecture,
    generate_coupling,
    generate_cyclic,
    generate_cyclic_sweep,
    generate_ecology_stress,
    generate_evolution,
    generate_evolution_evidence,
    generate_graded,
    generate_homeostasis,
    generate_invariance,
    generate_lifelikeness_gap,
    generate_lineage,
    generate_midrun_ablation,
    generate_orthogonal,
    generate_persistent_clusters,
    generate_phenotype,
    generate_proxy,
    generate_spatial,
    generate_timeseries,
    generate_trait_evolution,
)
from figures._shared import get_coupling_best  # noqa: F401  (re-export for back-compat)
from figures.fig_orthogonal import plot_violin_strip  # noqa: F401  (re-export for back-compat)


def main() -> None:
    """Run all figure generators with graceful skip on missing data."""
    from figures._shared import DATA_TSV, parse_tsv

    print("Generating paper figures...")

    print("Figure 1: Architecture diagram")
    generate_architecture()

    print("Figure 2: Time-series plot")
    if DATA_TSV.exists():
        data = parse_tsv(DATA_TSV)
        print(f"  Parsed {len(data)} rows from {DATA_TSV.name}")
        generate_timeseries(data)
    else:
        print(f"  SKIP: {DATA_TSV} not found")

    print("Figure 3: Proxy control comparison")
    generate_proxy()

    print("Figure 4: Evolution strengthening")
    generate_evolution()

    print("Figure 5: Homeostasis trajectory")
    generate_homeostasis()

    print("Figure 6: Ablation distributions")
    generate_ablation_distributions()

    print("Figure 7: Graded ablation dose-response")
    generate_graded()

    print("Figure 8: Cyclic environment")
    generate_cyclic()

    print("Figure 9: Phenotype clustering")
    generate_phenotype()

    print("Figure 10: Coupling graph")
    generate_coupling()

    print("Figure 11: Spatial cohesion")
    generate_spatial()

    print("Figure 12: Lineage phylogeny")
    generate_lineage()

    print("Figure 13: Cyclic period sweep")
    generate_cyclic_sweep()

    print("Figure 14: Orthogonal metrics")
    generate_orthogonal()

    print("Figure 15: Evolution evidence")
    generate_evolution_evidence()

    print("Figure 16: Persistent clusters")
    generate_persistent_clusters()

    print("Figure 17: Mid-run ablation")
    generate_midrun_ablation()

    print("Figure 18: Implementation invariance")
    generate_invariance()

    print("Figure 19: Ecology stressor")
    generate_ecology_stress()

    print("Figure 20: Trait evolution / selection differential")
    generate_trait_evolution()

    print("Figure 21: Life-likeness gap (Phase 0 diagnostic)")
    generate_lifelikeness_gap()

    print("Figure 22: 8th Criterion — memory results")
    generate_criterion8()

    print("Done.")


if __name__ == "__main__":
    main()
