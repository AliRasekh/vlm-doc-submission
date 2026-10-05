"""Helpers for strict CLI JSON stdout behavior."""

from __future__ import annotations

import contextlib
import io
import json
import sys
from typing import Any, Iterator


REQUIRED_JSON_KEYS = (
    "status",
    "prediction",
    "max_num_cap",
    "actual_tile_count",
    "request_seconds",
    "request_timing_scope",
    "live_inference",
    "references_used_in_model",
)


@contextlib.contextmanager
def stdout_to_stderr() -> Iterator[None]:
    """Route print()/stdout notices to stderr (keeps --json stdout pure)."""
    with contextlib.redirect_stdout(sys.stderr):
        yield


def emit_json(obj: Any) -> None:
    """Write exactly one JSON document to stdout."""
    sys.stdout.write(json.dumps(obj, indent=2, default=str))
    sys.stdout.write("\n")
    sys.stdout.flush()


def parse_cli_json_stdout(stdout: str) -> dict[str, Any]:
    """Strict parse: entire stdout must be one JSON object (no brace hunting)."""
    if stdout is None:
        raise ValueError("CLI stdout is None")
    text = stdout.strip()
    if not text:
        raise ValueError("CLI stdout is empty")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"CLI stdout is not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"CLI JSON root must be an object, got {type(obj).__name__}")
    missing = [k for k in REQUIRED_JSON_KEYS if k not in obj]
    if missing:
        raise ValueError(f"CLI JSON missing keys: {missing}")
    return obj


def simulate_noisy_stdout_json(payload: dict[str, Any], notice: str) -> str:
    """Test helper: notice + nested JSON string (must NOT be accepted by strict parse)."""
    return f"{notice}\n{json.dumps(payload, indent=2)}\n"
