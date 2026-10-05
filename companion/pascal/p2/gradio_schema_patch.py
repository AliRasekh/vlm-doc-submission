"""Targeted Gradio schema patch for Image/File event API info crash.

Installed Gradio 5.9.1 / gradio_client 1.5.2 build JSON schemas where
``additionalProperties`` can be the boolean ``True``. Upstream then:

1. ``get_type()`` does ``\"const\" in schema`` → TypeError on bool
2. ``_json_schema_to_python_type()`` raises ``APIInfoParseError`` on bool

This module patches both helpers to treat bare bool schemas as ``Any``,
restoring ``Blocks.get_api_info()`` without blanking the whole API schema.
Do not upgrade the core ML stack for this.
"""

from __future__ import annotations

from typing import Any


_PATCHED = False


def apply_gradio_schema_patch() -> dict[str, Any]:
    global _PATCHED
    if _PATCHED:
        return {"applied": True, "already": True}
    import gradio_client.utils as utils

    original_get_type = utils.get_type
    original_json_schema = utils._json_schema_to_python_type

    def get_type_safe(schema: Any) -> str:
        if isinstance(schema, bool):
            return "Any"
        if not isinstance(schema, dict):
            return "Any"
        return original_get_type(schema)

    def json_schema_to_python_type_safe(schema: Any, defs: Any) -> str:
        if isinstance(schema, bool):
            return "Any"
        if schema is None:
            return "Any"
        if not isinstance(schema, dict):
            return "Any"
        return original_json_schema(schema, defs)

    utils.get_type = get_type_safe  # type: ignore[assignment]
    utils._json_schema_to_python_type = json_schema_to_python_type_safe  # type: ignore[assignment]
    # Public wrapper used by Blocks.get_api_info
    if hasattr(utils, "json_schema_to_python_type"):
        def public_wrapper(schema: Any) -> str:
            if isinstance(schema, bool):
                return "Any"
            if not isinstance(schema, dict):
                return "Any"
            return json_schema_to_python_type_safe(schema, schema.get("$defs"))

        utils.json_schema_to_python_type = public_wrapper  # type: ignore[assignment]

    _PATCHED = True
    return {
        "applied": True,
        "already": False,
        "modules": [
            "gradio_client.utils.get_type",
            "gradio_client.utils._json_schema_to_python_type",
            "gradio_client.utils.json_schema_to_python_type",
        ],
        "reason": (
            "additionalProperties=True (bool) crashes upstream get_type / "
            "_json_schema_to_python_type on Gradio 5.9.1 / gradio_client 1.5.2"
        ),
        "gradio": _pkg_ver("gradio"),
        "gradio_client": _pkg_ver("gradio_client"),
    }


def _pkg_ver(name: str) -> str | None:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", None)
    except Exception:
        return None
