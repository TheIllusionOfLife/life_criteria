from __future__ import annotations

import importlib
import sys
import types


def _load_module():
    fake = types.SimpleNamespace(version=lambda: "test")
    original = sys.modules.get("life_criteria")
    sys.modules["life_criteria"] = fake
    try:
        mod = importlib.import_module("scripts.experiment_candidateB_relaxed_cap")
        return importlib.reload(mod)
    finally:
        if original is None:
            del sys.modules["life_criteria"]
        else:
            sys.modules["life_criteria"] = original


def test_relaxed_cap_uses_matched_10k_protocol():
    mod = _load_module()
    assert mod.STEPS == 10_000
    assert mod.SAMPLE_EVERY == 100
    assert mod._ABLATION_STEP == mod.STEPS // 2

    assert mod._FAMINE_OVERRIDES["environment_shift_step"] == mod.STEPS * 3 // 10
    assert mod._BOOM_BUST_OVERRIDES["environment_cycle_period"] == mod.STEPS // 4
    assert mod._SEASONAL_OVERRIDES["environment_cycle_period"] == mod.STEPS // 10
