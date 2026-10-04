"""Deprecated shim. The data pipeline now lives in ``src.preprocessing``.

This module used to re-read the CSV with latin-1 and mix cleaning with the
split. That logic moved to :mod:`src.preprocessing`, which loads through the
validated UTF-8 loader in :mod:`src.dataset` instead. Import from there.

Kept temporarily so older imports keep working:

    from src.preprocessing import clean_dataset, split_dataset, save_splits
"""

from __future__ import annotations

import warnings

from .preprocessing import (  # noqa: F401  (re-exported for backwards compatibility)
    build_splits,
    clean_dataset,
    load_splits,
    save_splits,
    split_dataset,
)

warnings.warn(
    "src.data_loader is deprecated; import from src.preprocessing instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-exported on purpose: this module exists only to keep the old import path
# working. __all__ states that intent so the names do not read as unused.
__all__ = [
    "clean_dataset",
    "split_dataset",
    "save_splits",
    "load_splits",
    "build_splits",
]
