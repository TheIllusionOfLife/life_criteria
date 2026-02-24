"""figures — subpackage containing one module per figure family.

Import any generate_X function directly:
    from figures import generate_timeseries
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# Ensure FIG_DIR exists before any figure module tries to write to it
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_FIG_DIR = _PROJECT_ROOT / "paper" / "figures"
_FIG_DIR.mkdir(parents=True, exist_ok=True)

from figures.fig_ablation import generate_ablation_distributions
from figures.fig_architecture import generate_architecture
from figures.fig_coupling import generate_coupling
from figures.fig_cyclic import generate_cyclic, generate_cyclic_sweep
from figures.fig_ecology_stress import generate_ecology_stress
from figures.fig_evolution import generate_evolution
from figures.fig_evolution_evidence import generate_evolution_evidence
from figures.fig_graded import generate_graded
from figures.fig_homeostasis import generate_homeostasis
from figures.fig_invariance import generate_invariance
from figures.fig_lineage import generate_lineage
from figures.fig_midrun import generate_midrun_ablation
from figures.fig_orthogonal import generate_orthogonal
from figures.fig_phenotype import generate_persistent_clusters, generate_phenotype
from figures.fig_proxy import generate_proxy
from figures.fig_spatial import generate_spatial
from figures.fig_timeseries import generate_timeseries
from figures.fig_trait_evolution import generate_trait_evolution

__all__ = [
    "generate_timeseries",
    "generate_architecture",
    "generate_proxy",
    "generate_evolution",
    "generate_homeostasis",
    "generate_ablation_distributions",
    "generate_graded",
    "generate_cyclic",
    "generate_cyclic_sweep",
    "generate_phenotype",
    "generate_persistent_clusters",
    "generate_coupling",
    "generate_spatial",
    "generate_lineage",
    "generate_orthogonal",
    "generate_evolution_evidence",
    "generate_midrun_ablation",
    "generate_invariance",
    "generate_ecology_stress",
    "generate_trait_evolution",
]
