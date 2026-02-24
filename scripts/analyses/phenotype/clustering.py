"""Phenotype k-means clustering and temporal persistence utilities."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

PERSISTENCE_CLAIM_THRESHOLD = 0.30


def persistence_claim_gate(ari: float, threshold: float = PERSISTENCE_CLAIM_THRESHOLD) -> bool:
    """Return True when ARI meets the threshold for stronger persistence claims."""
    return bool(ari >= threshold)


def cluster_phenotypes(traits: np.ndarray, max_k: int = 5) -> dict:
    """Perform k-means clustering on trait vectors, selecting k by silhouette score."""
    if len(traits) < 4:
        return {"n_clusters": 0, "error": "insufficient data"}

    scaler = StandardScaler()
    scaled = scaler.fit_transform(traits)

    # Try k=2..max_k and pick best silhouette
    best_k = 2
    best_score = -1.0
    best_model = None
    for k in range(2, min(max_k + 1, len(traits))):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        cur_labels = km.fit_predict(scaled)
        score = silhouette_score(scaled, cur_labels)
        if score > best_score:
            best_score = score
            best_k = k
            best_model = km

    labels = best_model.predict(scaled)

    # Compute per-cluster trait means
    cluster_profiles = []
    trait_names = [
        "energy_mean",
        "waste_mean",
        "boundary_mean",
        "genome_diversity",
        "mean_generation",
    ]
    for c in range(best_k):
        mask = labels == c
        profile = {
            "cluster_id": c,
            "count": int(mask.sum()),
        }
        for i, name in enumerate(trait_names):
            profile[name] = round(float(traits[mask, i].mean()), 4)
            profile[f"{name}_std"] = round(float(traits[mask, i].std()), 4)
        cluster_profiles.append(profile)

    return {
        "n_clusters": best_k,
        "silhouette_score": round(float(best_score), 4),
        "cluster_profiles": cluster_profiles,
        "labels": [int(label) for label in labels],
        "trait_names": trait_names,
        "traits": [[round(float(v), 4) for v in row] for row in traits],
    }


def _compute_clustering_ari(
    traits_a: np.ndarray, traits_b: np.ndarray, k: int = 2
) -> tuple[float, np.ndarray, np.ndarray]:
    """Standardize and cluster two sets of traits, then compute ARI."""
    scaled_a = StandardScaler().fit_transform(traits_a)
    km_a = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels_a = km_a.fit_predict(scaled_a)

    scaled_b = StandardScaler().fit_transform(traits_b)
    km_b = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels_b = km_b.fit_predict(scaled_b)

    ari = adjusted_rand_score(labels_a, labels_b)
    return float(ari), labels_a, labels_b


def _summarize_window(
    labels: np.ndarray,
    traits_raw: np.ndarray,
    n_total: int | None,
    k: int,
    trait_names: list[str],
    prefix: str = "",
    include_extra: bool = True,
) -> dict:
    """Summarize clustering for a time window."""
    profiles = []
    proportions = []
    for c in range(k):
        mask = labels == c
        count = int(mask.sum())
        proportions.append(round(count / len(labels), 4))
        profile = {"cluster_id": c, "count": count}
        for i, name in enumerate(trait_names):
            mean_val = float(traits_raw[mask, i].mean()) if count > 0 else 0.0
            profile[f"{prefix}{name}"] = round(mean_val, 4)
        profiles.append(profile)

    res = {
        "n_clusters": k,
        "cluster_proportions": proportions,
        "cluster_profiles": profiles,
    }

    if include_extra:
        if n_total is not None:
            res["n_total_organisms"] = n_total
        res["n_shared_organisms"] = len(labels)
        # Silhouette score only if we have enough samples
        if len(traits_raw) > k and len(np.unique(labels)) > 1:
            scaled = StandardScaler().fit_transform(traits_raw)
            sil = silhouette_score(scaled, labels)
        else:
            sil = 0.0
        res["silhouette_score"] = round(float(sil), 4)

    return res
