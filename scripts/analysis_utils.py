"""Shared helpers for analysis scripts.

.. deprecated::
    Inline ``json.load(open(path))`` or import from the relevant
    ``analyses/`` subpackage instead. This stub will be removed in a
    future cleanup.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

warnings.warn(
    "analysis_utils is deprecated; inline the load() call or use the analyses/ package.",
    DeprecationWarning,
    stacklevel=2,
)


def load(path: Path) -> list[dict]:
    """Load a JSON array from disk."""
    with open(path) as f:
        return json.load(f)
