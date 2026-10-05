"""Regression: strict CLI JSON parsing + nested/noisy stdout rejection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "companion" / "pascal"))

from p2.cli_json import (  # noqa: E402
    REQUIRED_JSON_KEYS,
    parse_cli_json_stdout,
    simulate_noisy_stdout_json,
)
from p2.presets import get_preset, match_preset_references, sha256_file  # noqa: E402


def _sample_payload(**overrides):
    base = {
        "status": "ok",
        "prediction": "USMM 1/95-6/95, 12-Month Data",
        "max_num_cap": 12,
        "actual_tile_count": 13,
        "request_seconds": 1.23,
        "request_timing_scope": "image_open_through_preprocess_transfer_generate_decode_cuda_synchronized",
        "live_inference": True,
        "references_used_in_model": False,
        "nested": {"inner": {"prediction": "SHOULD_NOT_BE_SELECTED"}, "x": [1, 2]},
    }
    base.update(overrides)
    return base


def test_strict_parse_accepts_pure_json():
    payload = _sample_payload()
    text = json.dumps(payload, indent=2) + "\n"
    obj = parse_cli_json_stdout(text)
    assert obj["prediction"] == payload["prediction"]
    for k in REQUIRED_JSON_KEYS:
        assert k in obj


def test_strict_parse_rejects_flashattn_prefix_and_nested_brace_hunting():
    payload = _sample_payload()
    noisy = simulate_noisy_stdout_json(
        payload, "FlashAttention2 is not installed."
    )
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_cli_json_stdout(noisy)
    # Legacy rfind("{") points at the innermost nested object, not the root —
    # demonstrating why brace-hunting is invalid for nested JSON.
    brace = noisy.rfind("{")
    assert '"SHOULD_NOT_BE_SELECTED"' in noisy[brace : brace + 80]
    with pytest.raises(json.JSONDecodeError):
        json.loads(noisy[brace:])
    # Two JSON values / notice prefix must not become a soft fallback.
    with pytest.raises(ValueError):
        parse_cli_json_stdout(
            '{"nested": {"prediction": "wrong"}}\n' + json.dumps(payload)
        )


def test_strict_parse_rejects_missing_keys():
    bad = {"status": "ok", "prediction": "x"}
    with pytest.raises(ValueError, match="missing keys"):
        parse_cli_json_stdout(json.dumps(bad))


def test_match_preset_by_content_hash_not_path(tmp_path):
    p = get_preset("exact_success")
    if not p["image_exists"]:
        pytest.skip("development images not materialized")
    copy = tmp_path / "cached_copy.png"
    copy.write_bytes(Path(p["image_path"]).read_bytes())
    assert sha256_file(copy) == p["image_sha256"]
    matched = match_preset_references(image_path=copy, question=p["question"])
    assert matched is not None
    assert matched["id"] == "exact_success"
    assert match_preset_references(
        image_path=copy, question="different question"
    ) is None
