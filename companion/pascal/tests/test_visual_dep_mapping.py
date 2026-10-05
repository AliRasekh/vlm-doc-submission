"""CPU tests for visual-dependence mapping rules (synthetic fixtures)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "companion" / "pascal"))

from visual_dep.freeze_mapping import (  # noqa: E402
    build_mapping,
    select_mapping_from_rows,
)


def _synth_rows():
    # Two aspect groups; docs/hashes differ so exclusions are testable.
    return [
        {
            "question_id": 1,
            "doc_id": "A",
            "image_relpath": "a1.png",
            "image_sha256_png": "hash1",
            "width": 100,
            "height": 100,
        },
        {
            "question_id": 2,
            "doc_id": "A",
            "image_relpath": "a2.png",
            "image_sha256_png": "hash2",
            "width": 100,
            "height": 100,
        },
        {
            "question_id": 3,
            "doc_id": "B",
            "image_relpath": "b1.png",
            "image_sha256_png": "hash3",
            "width": 200,
            "height": 100,
        },
        {
            "question_id": 4,
            "doc_id": "C",
            "image_relpath": "c1.png",
            "image_sha256_png": "hash4",
            "width": 100,
            "height": 100,
        },
    ]


def test_mapping_excludes_same_doc_and_same_image():
    payload = select_mapping_from_rows(_synth_rows(), seed=2026)
    assert payload["n_queries"] == 4
    assert payload["mapping_sha256"]
    for _key, m in payload["mapping"].items():
        assert m["query_qid"] != m["donor_qid"]
        assert m["query_doc_id"] != m["donor_doc_id"]
        assert m["query_image_sha256_png"] != m["donor_image_sha256_png"]
        assert m["resized_donor_to_query_geometry"] is False


def test_mapping_deterministic():
    a = select_mapping_from_rows(_synth_rows(), seed=2026)
    b = select_mapping_from_rows(_synth_rows(), seed=2026)
    assert a["mapping_sha256"] == b["mapping_sha256"]
    assert a["mapping"] == b["mapping"]


@pytest.mark.integration
def test_full_dev_mapping_requires_images():
    """Full 208-Q mapping needs materialized data/images/docvqa_dev/."""
    man = REPO / "data/manifests/docvqa_dev_v1.json"
    sample = json_load_first_image(man)
    if sample is None or not sample.is_file():
        pytest.skip("development images not materialized")
    payload = build_mapping(2026)
    assert payload["n_queries"] == 208


def json_load_first_image(man_path: Path) -> Path | None:
    import json

    if not man_path.is_file():
        return None
    man = json.loads(man_path.read_text())
    ex = (man.get("examples") or [None])[0]
    if not ex:
        return None
    return REPO / ex["image_relpath"]
