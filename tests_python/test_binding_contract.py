"""Binding contract tests: verify the PyO3 life_criteria module schema.

These tests use the real Rust binary (no mocks) to validate that the
JSON schema emitted by run_experiment_json() remains stable across refactors.
"""

from __future__ import annotations

import json

import life_criteria
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_OVERRIDE = {
    "num_organisms": 2,
    "agents_per_organism": 5,
    "world_size": 20.0,
    "seed": 42,
    "growth_maturation_steps": 10,
}


def _make_config(**overrides) -> str:
    """Build a minimal JSON config suitable for fast test runs."""
    base = json.loads(life_criteria.default_config_json())
    base.update(_MINIMAL_OVERRIDE)
    base.update(overrides)
    return json.dumps(base)


# ---------------------------------------------------------------------------
# default_config_json / validate_config_json round-trip
# ---------------------------------------------------------------------------


def test_default_config_round_trips_through_validate():
    """default_config_json() output must be accepted by validate_config_json()."""
    config_json = life_criteria.default_config_json()
    assert life_criteria.validate_config_json(config_json) is True


def test_default_config_is_valid_json():
    cfg = json.loads(life_criteria.default_config_json())
    assert isinstance(cfg, dict)
    assert cfg["num_organisms"] > 0
    assert cfg["world_size"] > 0


# ---------------------------------------------------------------------------
# run_experiment_json schema
# ---------------------------------------------------------------------------


def test_run_experiment_json_schema_version():
    """RunSummary must carry schema_version == 1."""
    result = json.loads(life_criteria.run_experiment_json(_make_config(), 10, 5))
    assert result["schema_version"] == 1


def test_run_experiment_json_required_top_level_fields():
    """All expected top-level keys must be present."""
    result = json.loads(life_criteria.run_experiment_json(_make_config(), 10, 5))
    required = {
        "schema_version",
        "steps",
        "sample_every",
        "final_alive_count",
        "samples",
        "lifespans",
        "total_reproduction_events",
        "lineage_events",
    }
    assert required.issubset(result.keys())


def test_run_experiment_json_types():
    """Top-level field types must match expected Python types."""
    result = json.loads(life_criteria.run_experiment_json(_make_config(), 10, 5))
    assert isinstance(result["steps"], int)
    assert isinstance(result["sample_every"], int)
    assert isinstance(result["final_alive_count"], int)
    assert isinstance(result["samples"], list)
    assert isinstance(result["lifespans"], list)
    assert isinstance(result["total_reproduction_events"], int)
    assert isinstance(result["lineage_events"], list)


def test_run_experiment_json_sample_fields():
    """Each StepMetrics sample must contain the expected metric fields."""
    result = json.loads(life_criteria.run_experiment_json(_make_config(), 10, 5))
    assert result["samples"], "Expected at least one sample"
    sample = result["samples"][0]
    required_sample_keys = {
        "step",
        "energy_mean",
        "waste_mean",
        "boundary_mean",
        "alive_count",
        "resource_total",
        "birth_count",
        "death_count",
        "population_size",
        "mean_generation",
        "mean_genome_drift",
        "agent_id_exhaustion_events",
        "energy_std",
        "waste_std",
        "boundary_std",
        "mean_age",
        "internal_state_mean",
        "internal_state_std",
        "genome_diversity",
        "max_generation",
        "maturity_mean",
        "spatial_cohesion_mean",
    }
    assert required_sample_keys.issubset(sample.keys()), (
        f"Missing keys: {required_sample_keys - sample.keys()}"
    )


def test_run_experiment_json_sample_count():
    """Number of samples must match ceil(steps / sample_every)."""
    steps, sample_every = 20, 5
    result = json.loads(life_criteria.run_experiment_json(_make_config(), steps, sample_every))
    expected = (steps + sample_every - 1) // sample_every
    assert len(result["samples"]) == expected


def test_run_experiment_json_steps_field_matches_argument():
    result = json.loads(life_criteria.run_experiment_json(_make_config(), 15, 5))
    assert result["steps"] == 15
    assert result["sample_every"] == 5


def test_validate_config_json_rejects_oversized_world():
    """validate_config_json() must reject world_size > MAX_WORLD_SIZE (2048.0)."""
    cfg = json.loads(life_criteria.default_config_json())
    cfg["world_size"] = 99_999.0  # exceeds MAX_WORLD_SIZE — caught at config validation layer
    with pytest.raises(Exception, match="world_size"):
        life_criteria.validate_config_json(json.dumps(cfg))
