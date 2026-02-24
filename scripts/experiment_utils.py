"""Compatibility shim for legacy imports.

.. deprecated::
    Import directly from ``experiment_common`` instead. This shim will be
    removed in a future cleanup. Migrate: replace
    ``from experiment_utils import X`` with ``from experiment_common import X``.
"""

import warnings

warnings.warn(
    "experiment_utils is deprecated; import from experiment_common directly.",
    DeprecationWarning,
    stacklevel=2,
)

from experiment_common import CONDITIONS, log, make_config, safe_path
from experiment_common import run_single as _run_single_common

__all__ = ["CONDITIONS", "log", "make_config", "run_single", "safe_path"]


def run_single(seed: int, steps: int, sample_every: int, *override_dicts: dict) -> dict:
    """Run a single experiment with merged overrides.

    This wrapper preserves the historical call shape used by older scripts.
    """
    merged_overrides = {}
    for overrides in override_dicts:
        merged_overrides.update(overrides)
    return _run_single_common(seed, merged_overrides, steps=steps, sample_every=sample_every)
