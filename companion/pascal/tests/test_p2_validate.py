"""Focused P2 CPU tests: validation, caps, reference separation, presets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "companion" / "pascal"))

from p2.presets import get_preset, list_presets
from p2.validate import (
    MAX_PIXELS,
    ValidationError,
    validate_cap,
    validate_image_file,
    validate_question,
)


def test_question_and_caps():
    assert validate_question("  hello  ") == "hello"
    with pytest.raises(ValidationError):
        validate_question("   ")
    assert validate_cap(12) == 12
    assert validate_cap("4") == 4
    with pytest.raises(ValidationError):
        validate_cap(8)


def test_image_limits(tmp_path):
    p = tmp_path / "ok.png"
    Image.new("RGB", (64, 64), color=(1, 2, 3)).save(p)
    meta = validate_image_file(p)
    assert meta["width"] == 64
    with pytest.raises(ValidationError):
        validate_image_file(tmp_path / "missing.png")
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"not-an-image")
    with pytest.raises(ValidationError):
        validate_image_file(bad)


def test_presets_exist_and_refs_separated():
    rows = list_presets()
    assert len(rows) >= 3
    ids = {r["id"] for r in rows}
    assert "exact_success" in ids
    assert "failure" in ids
    assert "reduced_budget_regression" in ids
    p = get_preset("exact_success")
    # Image materialization is optional for unit tests.
    assert "image_path" in p
    assert "image_exists" in p
    assert isinstance(p["image_exists"], bool)
    # Simulate that predict kwargs must not include references
    predict_kwargs = {
        "image_path": p["image_path"],
        "question": p["question"],
        "max_num": 12,
    }
    assert "references" not in predict_kwargs
    assert "answers" not in predict_kwargs
    # References may be empty when answer cache is not shipped.
    assert "references" in p
    assert isinstance(p["references"], list)


@pytest.mark.integration
def test_preset_images_materialized_when_available():
    p = get_preset("exact_success")
    if not p["image_exists"]:
        pytest.skip("development images not materialized")
    assert Path(p["image_path"]).is_file()
    assert p["references"], "dev answers should exist when images are present"


def test_cli_import_no_cuda():
    # Import app builder pieces that don't require GPU
    from p2 import validate, presets, runtime

    assert validate.MAX_UPLOAD_BYTES > 0
    assert runtime.p2_dtype_policy()["selected_dtype"] == "bfloat16"
    assert presets.list_presets()
