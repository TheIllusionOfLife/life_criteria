# configs/

Versioned experiment configuration files. These values were calibrated via parameter sweeps
and are the single source of truth — do not edit without re-running the originating sweep.

| File | Provenance | Script |
|------|-----------|--------|
| `tuned_baseline.json` | Baseline parameters swept on 2026-02-12, calibration set seeds 0–99 | `scripts/param_sweep_thresholds.py` |

## Adding a new config

1. Run the relevant calibration script (e.g. `param_sweep_thresholds.py`).
2. Record the chosen values in a new JSON file here with a `_provenance` key.
3. Update the table above.
4. Update the loading code in `scripts/experiment_common.py` or the relevant analysis script.
