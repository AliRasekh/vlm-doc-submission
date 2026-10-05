"""Unit tests for P3 analysis integrity and eval-cache validation.

Default tests use synthetic fixtures only (no private run artifacts).

Integration tests requiring local Pascal caches are marked
``pytest.mark.integration`` and skip when artifacts are absent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "companion" / "pascal"))

from p3.analyze import (  # noqa: E402
    EvalIntegrityError,
    load_predictions_jsonl,
    load_requested_qids,
    score_against_requested,
)
from p3.run_main import (  # noqa: E402
    EXPECTED_CONFIG_SHA256,
    EXPECTED_INSTRUCTION,
    EXPECTED_MANIFEST_CONTENT_SHA256,
    EXPECTED_MODEL_ID,
    EXPECTED_MODEL_REVISION,
    N_DEV_EXPECTED,
    validate_eval_cache,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _minimal_fingerprint(**overrides) -> dict:
    fp = {
        "model_id": EXPECTED_MODEL_ID,
        "model_revision": EXPECTED_MODEL_REVISION,
        "manifest_content_sha256": EXPECTED_MANIFEST_CONTENT_SHA256,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "instruction": EXPECTED_INSTRUCTION,
        "max_new_tokens": 64,
        "do_sample": False,
        "prefer_bf16": True,
        "processor_settings": {"max_num": 12, "use_thumbnail": True},
        "run_fingerprint_sha256": "a" * 64,
    }
    fp.update(overrides)
    return fp


def _complete_summary(*, n_ok: int = 208, adapter: bool = False) -> dict:
    fp = _minimal_fingerprint()
    if adapter:
        fp["adapter_weights_sha256"] = {
            "adapter_model.safetensors": "b" * 64,
        }
        fp["adapter_dir"] = "synthetic/adapter"
        fp["train_update"] = 200
    return {
        "model_id": EXPECTED_MODEL_ID,
        "model_revision": EXPECTED_MODEL_REVISION,
        "manifest_content_sha256": EXPECTED_MANIFEST_CONTENT_SHA256,
        "scores": {
            "n_requested": N_DEV_EXPECTED,
            "n_completed_ok": n_ok,
            "n_failed": 0,
            "n_missing": N_DEV_EXPECTED - n_ok,
        },
        "run_fingerprint": fp,
    }


def test_score_missing_and_failed_count_as_zero(tmp_path: Path):
    requested = [1, 2, 3, 4]
    rows = [
        {
            "question_id": 1,
            "status": "ok",
            "metrics": {"anls_normalized_strict_v2": 1.0, "exact_match": 1.0},
        },
        {
            "question_id": 2,
            "status": "error",
            "metrics": {"anls_normalized_strict_v2": 0.9, "exact_match": 1.0},
        },
        {
            "question_id": 4,
            "status": "ok",
            "metrics": {"anls_normalized_strict_v2": 0.5, "exact_match": 0.0},
        },
    ]
    p = tmp_path / "preds.jsonl"
    _write_jsonl(p, rows)
    preds = load_predictions_jsonl(p, requested_qids=requested)
    scored = score_against_requested(preds, requested)
    assert scored["n_ok"] == 2
    assert scored["n_failed"] == 1
    assert scored["n_missing"] == 1
    assert scored["mean_anls"] == pytest.approx(1.5 / 4)
    assert scored["mean_em"] == pytest.approx(1.0 / 4)


def test_reject_duplicate_and_unexpected_qids(tmp_path: Path):
    requested = [10, 11]
    p = tmp_path / "dup.jsonl"
    _write_jsonl(
        p,
        [
            {
                "question_id": 10,
                "status": "ok",
                "metrics": {"anls_normalized_strict_v2": 1.0, "exact_match": 1.0},
            },
            {
                "question_id": 10,
                "status": "ok",
                "metrics": {"anls_normalized_strict_v2": 0.0, "exact_match": 0.0},
            },
        ],
    )
    with pytest.raises(EvalIntegrityError, match="duplicate"):
        load_predictions_jsonl(p, requested_qids=requested)

    p2 = tmp_path / "unexpected.jsonl"
    _write_jsonl(
        p2,
        [
            {
                "question_id": 10,
                "status": "ok",
                "metrics": {"anls_normalized_strict_v2": 1.0, "exact_match": 1.0},
            },
            {
                "question_id": 99,
                "status": "ok",
                "metrics": {"anls_normalized_strict_v2": 1.0, "exact_match": 1.0},
            },
        ],
    )
    with pytest.raises(EvalIntegrityError, match="unexpected"):
        load_predictions_jsonl(p2, requested_qids=requested)


def test_primary_metrics_use_requested_set_not_intersection(tmp_path: Path):
    requested = [1, 2, 3]
    rows = [
        {
            "question_id": 1,
            "status": "ok",
            "metrics": {"anls_normalized_strict_v2": 1.0, "exact_match": 1.0},
        },
        {
            "question_id": 2,
            "status": "ok",
            "metrics": {"anls_normalized_strict_v2": 1.0, "exact_match": 1.0},
        },
    ]
    p = tmp_path / "preds.jsonl"
    _write_jsonl(p, rows)
    preds = load_predictions_jsonl(p, requested_qids=requested)
    scored = score_against_requested(preds, requested)
    assert scored["mean_anls"] == pytest.approx(2.0 / 3)


def test_validate_eval_cache_rejects_summary_only(tmp_path: Path):
    summary = tmp_path / "dev_eval_summary.json"
    jsonl = tmp_path / "dev_eval.jsonl"
    summary.write_text(json.dumps(_complete_summary()))
    status, msg = validate_eval_cache(
        tag="fresh_base",
        summary_path=summary,
        jsonl_path=jsonl,
        expected_adapter_dir=None,
        expected_train_update=None,
    )
    assert status == "incompatible"
    assert "without predictions" in msg


def test_validate_eval_cache_rejects_missing_fingerprints(tmp_path: Path):
    summary = tmp_path / "dev_eval_summary.json"
    jsonl = tmp_path / "dev_eval.jsonl"
    summary.write_text(
        json.dumps(
            {
                "model_id": EXPECTED_MODEL_ID,
                "model_revision": EXPECTED_MODEL_REVISION,
                "scores": {
                    "n_requested": 208,
                    "n_completed_ok": 0,
                    "n_failed": 0,
                    "n_missing": 208,
                },
                "run_fingerprint": {"model_id": EXPECTED_MODEL_ID},
            }
        )
    )
    jsonl.write_text("")
    status, msg = validate_eval_cache(
        tag="fresh_base",
        summary_path=summary,
        jsonl_path=jsonl,
        expected_adapter_dir=None,
        expected_train_update=None,
    )
    assert status == "incompatible"
    assert "missing required fingerprint field" in msg


def test_validate_eval_cache_rejects_mismatched_adapter_hash(tmp_path: Path):
    """Synthetic fixture: fingerprint adapter hash disagrees with on-disk meta."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"fake-weights")
    (adapter_dir / "vlm_doc_checkpoint_meta.json").write_text(
        json.dumps({"adapter_weight_sha256": {"adapter_model.safetensors": "c" * 64}})
    )

    requested = load_requested_qids()
    rows = []
    fp = _minimal_fingerprint(
        adapter_weights_sha256={"adapter_model.safetensors": "d" * 64},
        adapter_dir=str(adapter_dir),
        train_update=200,
    )
    for qid in requested:
        rows.append(
            {
                "question_id": qid,
                "status": "ok",
                "prediction": "x",
                "metrics": {
                    "anls_normalized_strict_v2": 0.0,
                    "exact_match": 0.0,
                },
                "run_fingerprint": fp,
            }
        )
    summary = _complete_summary(adapter=True)
    summary["run_fingerprint"] = fp
    summary["scores"] = {
        "n_requested": N_DEV_EXPECTED,
        "n_completed_ok": N_DEV_EXPECTED,
        "n_failed": 0,
        "n_missing": 0,
    }
    sp = tmp_path / "dev_eval_summary.json"
    jp = tmp_path / "dev_eval.jsonl"
    sp.write_text(json.dumps(summary))
    _write_jsonl(jp, rows)

    status, msg = validate_eval_cache(
        tag="seed_synth",
        summary_path=sp,
        jsonl_path=jp,
        expected_adapter_dir=adapter_dir,
        expected_train_update=200,
    )
    assert status == "incompatible"
    assert "adapter weight sha mismatch" in msg


def test_load_requested_qids_from_manifest():
    qids = load_requested_qids()
    assert len(qids) == 208
    assert len(set(qids)) == 208


@pytest.mark.integration
def test_real_caches_still_validate():
    """Requires local Pascal P3 eval caches under companion/pascal/artifacts/p3/."""
    from p3.paths import P3_ART, seed_dirs

    base_summary = P3_ART / "eval_fresh_base/dev_eval_summary.json"
    if not base_summary.is_file():
        pytest.skip("P3 eval caches not present (integration only)")
    st, _ = validate_eval_cache(
        tag="fresh_base",
        summary_path=base_summary,
        jsonl_path=P3_ART / "eval_fresh_base/dev_eval.jsonl",
        expected_adapter_dir=None,
        expected_train_update=None,
    )
    assert st == "ok_complete"
    for seed in (42, 43, 44):
        d = seed_dirs(seed)
        if not d["eval_summary"].is_file():
            pytest.skip(f"missing seed_{seed} eval cache")
        st, _ = validate_eval_cache(
            tag=f"seed_{seed}",
            summary_path=d["eval_summary"],
            jsonl_path=d["eval_jsonl"],
            expected_adapter_dir=d["checkpoint_root"] / "update_0200",
            expected_train_update=200,
        )
        assert st == "ok_complete"
