"""Model-family adapters for DocVQA evaluation."""

from __future__ import annotations

from .internvl import (
    LoadedInternVL,
    load_internvl,
    processor_settings_dict as internvl_processor_settings,
    run_one_internvl,
)

__all__ = [
    "LoadedInternVL",
    "load_internvl",
    "run_one_internvl",
    "internvl_processor_settings",
]
