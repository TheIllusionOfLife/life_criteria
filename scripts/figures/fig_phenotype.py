"""Figures 9 and 16: Phenotype clustering and persistent clusters."""

import numpy as np
from figures._shared import *


def generate_phenotype() -> None:
    """Figure 9: Phenotype clustering scatter plot."""
    analysis_path = PROJECT_ROOT / "experiments" / "phenotype_analysis.json"
    if not analysis_path.exists():
        print(f"  SKIP: {analysis_path} not found")
        return

    with open(analysis_path) as f:
        import json

        analysis = json.load(f)

    traits = np.array(analysis.get("traits", []))
    labels = np.array(analysis.get("labels", []))

    if traits.shape[0] < 4 or traits.shape[1] < 2:
        print("  SKIP: insufficient trait data for phenotype plot")
        return

    # Use PCA for 2D projection if >2 dimensions
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(traits)

    n_clusters = analysis.get("n_clusters", 2)
    cluster_colors = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7"]

    fig, ax = plt.subplots(figsize=(3.4, 3.0))

    for c in range(n_clusters):
        mask = labels == c
        ax.scatter(
            proj[mask, 0],
            proj[mask, 1],
            s=15,
            alpha=0.6,
            color=cluster_colors[c % len(cluster_colors)],
            label=f"Cluster {c} (n={mask.sum()})",
            edgecolors="none",
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var)")
    ax.legend(loc="best", fontsize=7, markerscale=1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_phenotype.pdf", format="pdf")
    plt.close(fig)
    print(f"  Saved {FIG_DIR / 'fig_phenotype.pdf'}")


def generate_persistent_clusters() -> None:
    """Figure 16: Per-organism phenotype clusters — PCA scatter at early vs late windows."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    analysis_path = PROJECT_ROOT / "experiments" / "phenotype_analysis.json"
    if not analysis_path.exists():
        print(f"  SKIP: {analysis_path} not found")
        return

    with open(analysis_path) as f:
        import json

        analysis = json.load(f)

    olp = analysis.get("organism_level_persistence")
    if not olp or "error" in olp:
        print("  SKIP: no organism_level_persistence data in phenotype_analysis.json")
        return

    ari = olp.get("adjusted_rand_index", 0.0)
    cluster_colors = ["#0072B2", "#D55E00"]

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.0))

    n_shared_early = olp["early_window"]["n_shared_organisms"]
    n_shared_late = olp["late_window"]["n_shared_organisms"]

    for panel_idx, (traits_key, labels_key, window_key, title) in enumerate(
        [
            (
                "early_traits",
                "early_labels",
                "early_window",
                f"(A) Early Window (n={n_shared_early})",
            ),
            ("late_traits", "late_labels", "late_window", f"(B) Late Window (n={n_shared_late})"),
        ]
    ):
        ax = axes[panel_idx]
        raw_traits = olp.get(traits_key, [])
        raw_labels = olp.get(labels_key, [])

        if len(raw_traits) < 4:
            ax.text(
                0.5,
                0.5,
                "Insufficient data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        traits = np.array(raw_traits)
        labels = np.array(raw_labels)

        scaled = StandardScaler().fit_transform(traits)
        pca = PCA(n_components=2, random_state=42)
        proj = pca.fit_transform(scaled)

        for c in sorted(set(raw_labels)):
            mask = labels == c
            ax.scatter(
                proj[mask, 0],
                proj[mask, 1],
                s=12,
                alpha=0.5,
                color=cluster_colors[c % len(cluster_colors)],
                label=f"Cluster {c} (n={mask.sum()})",
                edgecolors="none",
            )

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var)")
        ax.set_title(title, fontsize=9)
        ax.legend(loc="best", fontsize=6, markerscale=1.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Annotate cluster proportions and silhouette
        window = olp.get(window_key, {})
        proportions = window.get("cluster_proportions", [])
        sil = window.get("silhouette_score", 0.0)
        prop_text = ", ".join(f"{p:.0%}" for p in proportions) if proportions else ""
        ax.text(
            0.02,
            0.98,
            f"Prop: {prop_text}\nSil: {sil:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor="0.8",
                alpha=0.9,
            ),
        )

    fig.text(
        0.5,
        0.01,
        f"Organism-level ARI: {ari:.3f} (early vs late window)",
        ha="center",
        va="bottom",
        fontsize=8,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="#FFF9C4",
            edgecolor="0.8",
            alpha=0.9,
        ),
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(FIG_DIR / "fig_persistent_clusters.pdf", format="pdf")
    plt.close(fig)
    print(f"  Saved {FIG_DIR / 'fig_persistent_clusters.pdf'}")
