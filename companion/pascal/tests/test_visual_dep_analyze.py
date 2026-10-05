"""CPU tests for visual_dep analysis integrity and agreement rules."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "companion" / "pascal"))

from visual_dep.analyze import (  # noqa: E402
    EvalIntegrityError,
    load_predictions_jsonl,
    load_requested_qids,
    prediction_agreement,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_load_requested_qids_expects_208():
    qids = load_requested_qids()
    assert len(qids) == 208
    assert len(set(qids)) == 208


def test_reject_duplicate_and_unexpected_including_clean(tmp_path: Path):
    requested = [1, 2]
    p = tmp_path / "dup.jsonl"
    _write_jsonl(
        p,
        [
            {"question_id": 1, "status": "ok", "prediction": "a"},
            {"question_id": 1, "status": "ok", "prediction": "b"},
        ],
    )
    with pytest.raises(EvalIntegrityError, match="duplicate"):
        load_predictions_jsonl(p, requested_qids=requested)

    p2 = tmp_path / "unexpected.jsonl"
    _write_jsonl(
        p2,
        [
            {"question_id": 1, "status": "ok", "prediction": "a"},
            {"question_id": 99, "status": "ok", "prediction": "x"},
        ],
    )
    with pytest.raises(EvalIntegrityError, match="unexpected"):
        load_predictions_jsonl(p2, requested_qids=requested)


def test_missing_clean_record_does_not_agree_via_empty():
    # Both missing/error must not count as agreement.
    assert prediction_agreement(None, None) is None
    err = {"status": "error", "prediction": None}
    assert prediction_agreement(err, err) is None
    ok = {"status": "ok", "prediction": "hello"}
    assert prediction_agreement(ok, None) is None
    assert prediction_agreement(None, ok) is None
    assert prediction_agreement(ok, {"status": "ok", "prediction": "hello"}) is True
    assert prediction_agreement(ok, {"status": "ok", "prediction": "other"}) is False


def test_failed_and_missing_score_as_zero_over_requested(tmp_path: Path):
    from visual_dep.analyze import _anls, _em

    requested = [10, 11, 12]
    rows = [
        {
            "question_id": 10,
            "status": "ok",
            "metrics": {"anls_normalized_strict_v2": 1.0, "exact_match": 1.0},
        },
        {"question_id": 11, "status": "error", "metrics": {"anls_normalized_strict_v2": 1.0}},
    ]
    p = tmp_path / "preds.jsonl"
    _write_jsonl(p, rows)
    preds = load_predictions_jsonl(p, requested_qids=requested)
    assert _anls(preds.get(10)) == 1.0
    assert _anls(preds.get(11)) == 0.0
    assert _anls(preds.get(12)) == 0.0
    assert _em(preds.get(12)) == 0.0
