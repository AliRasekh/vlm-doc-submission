"""Resolve InternVL conversation templates from the pinned remote-code package."""

from __future__ import annotations

from typing import Any


def get_conv_template_fn(model: Any):
    """Import get_conv_template from the same remote-code package as the model."""
    base = model.__class__.__module__.rsplit(".", 1)[0]
    mod = __import__(f"{base}.conversation", fromlist=["get_conv_template"])
    return mod.get_conv_template


def get_conv_template(name: str, *, model: Any | None = None):
    if model is None:
        raise RuntimeError("model required to resolve InternVL conversation module")
    return get_conv_template_fn(model)(name)
