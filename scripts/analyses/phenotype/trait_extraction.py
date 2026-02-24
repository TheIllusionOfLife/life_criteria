"""Per-organism and per-seed trait extraction from experiment snapshots."""

from __future__ import annotations

import numpy as np


def extract_organism_traits(results: list[dict]) -> np.ndarray:
    """Extract per-seed population-level traits from final timestep samples.

    Returns array of shape (n_seeds, 5) with columns:
    [energy_mean, waste_mean, boundary_mean, genome_diversity, mean_generation]
    """
    traits = []
    for r in results:
        if "samples" not in r or not r["samples"]:
            continue
        final = r["samples"][-1]
        traits.append(
            [
                final.get("energy_mean", 0),
                final.get("waste_mean", 0),
                final.get("boundary_mean", 0),
                final.get("genome_diversity", 0),
                final.get("mean_generation", 0),
            ]
        )
    return np.array(traits) if traits else np.empty((0, 5))


def _extract_traits_at_step(results: list[dict], target_step: int) -> np.ndarray:
    """Extract trait vectors from the sample closest to target_step."""
    trait_names = [
        "energy_mean",
        "waste_mean",
        "boundary_mean",
        "genome_diversity",
        "mean_generation",
    ]
    traits = []
    for r in results:
        if "samples" not in r or not r["samples"]:
            continue
        # Find sample closest to target_step
        best = min(r["samples"], key=lambda s: abs(s.get("step", 0) - target_step))
        traits.append([best.get(name, 0) for name in trait_names])
    return np.array(traits) if traits else np.empty((0, 5))


def _collect_organism_traits(
    results: list[dict], frame_idx: int, trait_names: list[str]
) -> dict[tuple[int, int], list[float]]:
    """Collect per-organism traits from a specific snapshot frame index."""
    orgs: dict[tuple[int, int], list[float]] = {}
    for r in results:
        seed = r.get("seed", 0)
        frames = r.get("organism_snapshots", [])
        if frame_idx >= len(frames):
            continue
        frame = frames[frame_idx]
        for org in frame.get("organisms", []):
            stable_id = org.get("stable_id")
            if stable_id is None:
                continue
            key = (seed, stable_id)
            orgs[key] = [float(org.get(name, 0.0)) for name in trait_names]
    return orgs


def _extract_shared_traits(
    dict_a: dict[tuple[int, int], list[float]],
    dict_b: dict[tuple[int, int], list[float]],
) -> tuple[list[tuple[int, int]], np.ndarray | None, np.ndarray | None]:
    """Find shared organisms between two snapshots and return their traits."""
    shared_keys = sorted(set(dict_a.keys()) & set(dict_b.keys()))
    if len(shared_keys) < 4:
        return shared_keys, None, None
    traits_a = np.array([dict_a[k] for k in shared_keys])
    traits_b = np.array([dict_b[k] for k in shared_keys])
    return shared_keys, traits_a, traits_b
