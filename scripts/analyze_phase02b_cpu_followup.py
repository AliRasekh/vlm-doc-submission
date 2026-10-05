#!/usr/bin/env python3
"""CPU-only Phase 2B follow-up: validate paired comparison, formatting diagnostic,
output-behavior inspection, and gallery provenance.

Does not change primary metrics, predictions, prompts, splits, or run inference.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

DATASET_ID = "HuggingFaceM4/DocumentVQA"
DATASET_REV = "a44195f8f989c10d06ab4ccc98ffe9cdb6d928e4"
GALLERY_QIDS = [5158, 292, 297, 57368, 5263, 53842]
# Earlier incorrect gallery documentation (Phase 2A follow-up) for comparison
GALLERY_DOC_ERRORS = {
    5158: {"doc_id": 1565, "ucsf_document_id": "yjbj0227", "ucsf_document_page_no": "1"},
    5263: {"doc_id": 1585, "ucsf_document_id": "yjbj0227", "ucsf_document_page_no": "21"},
}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_last_by_qid(path: Path) -> dict[int, dict]:
    by: dict[int, dict] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            by[int(rec["question_id"])] = rec
    return by


def strip_one_terminal_period(normalized: str) -> str:
    """After normalize_text: remove at most one trailing ASCII '.' if present."""
    if normalized.endswith("."):
        return normalized[:-1]
    return normalized


def exact_match_diagnostic_terminal_period(
    prediction: str, references: list[str]
) -> float:
    """Diagnostic only — not a primary metric.

    Existing normalize (lower/strip/collapse) then strip at most one terminal
    ASCII full stop from BOTH prediction and each reference. Does not remove
    internal punctuation, decimals, %, units, articles, or arbitrary suffixes.
    """
    from vlm_doc.metrics import normalize_text, _valid_references

    pred_n = strip_one_terminal_period(normalize_text(prediction))
    refs = _valid_references(references)
    if not refs:
        return 0.0
    for r in refs:
        ref_n = strip_one_terminal_period(normalize_text(r))
        if pred_n == ref_n:
            return 1.0
    return 0.0


def ends_with_period_raw(prediction: str | None) -> bool:
    return bool(prediction) and str(prediction).endswith(".")


def percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def main() -> int:
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        print("ERROR: PYTHONNOUSERSITE=1 required", file=sys.stderr)
        return 2

    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))

    from vlm_doc.metrics import (
        anls_normalized_strict_v2,
        exact_match_single,
        normalize_text,
    )

    manifest = json.loads(Path("data/manifests/docvqa_dev_v1.json").read_text())
    smoke = json.loads(Path("data/manifests/docvqa_smoke_v1.json").read_text())
    qids = [int(x) for x in manifest["question_ids"]]
    assert len(qids) == 208 and len(set(qids)) == 208
    man_by = {int(e["question_id"]): e for e in manifest["examples"]}
    smoke_by = {int(e["question_id"]): e for e in smoke["examples"]}
    assert set(man_by) == set(qids)

    answers_blob = json.loads(Path("data/cache/docvqa_dev_v1_answers.json").read_text())
    answers_list = answers_blob["answers"]
    ans_by = {int(a["question_id"]): list(a["answers"]) for a in answers_list}
    assert set(ans_by) == set(qids), "answer sidecar coverage mismatch"
    assert len(ans_by) == 208

    m256 = load_last_by_qid(Path("results/dev_smolvlm256m.jsonl"))
    m500 = load_last_by_qid(Path("results/dev_smolvlm500m.jsonl"))
    assert set(m256) == set(qids), f"256M qids mismatch {len(m256)}"
    assert set(m500) == set(qids), f"500M qids mismatch {len(m500)}"
    assert len(m256) == 208 and len(m500) == 208

    # Question text consistency vs manifest
    for qid in qids:
        mq = man_by[qid]["question"]
        assert m256[qid].get("question") == mq, f"256 question mismatch {qid}"
        assert m500[qid].get("question") == mq, f"500 question mismatch {qid}"
        assert m256[qid].get("status") == "ok" and m500[qid].get("status") == "ok"

    prev = json.loads(Path("results/dev_256_vs_500_comparison.json").read_text())

    # --- Recompute primary metrics from predictions ---
    rows = []
    sum_pos = 0.0
    sum_neg = 0.0
    n_imp = n_reg = n_tie = 0
    for qid in qids:
        refs = ans_by[qid]
        p256 = m256[qid]["prediction"]
        p500 = m500[qid]["prediction"]
        s256 = anls_normalized_strict_v2(p256, refs)
        s500 = anls_normalized_strict_v2(p500, refs)
        e256 = exact_match_single(p256, refs)
        e500 = exact_match_single(p500, refs)
        d256 = exact_match_diagnostic_terminal_period(p256, refs)
        d500 = exact_match_diagnostic_terminal_period(p500, refs)
        delta = s500 - s256
        if delta > 1e-12:
            n_imp += 1
            sum_pos += delta
        elif delta < -1e-12:
            n_reg += 1
            sum_neg += delta
        else:
            n_tie += 1
        rows.append(
            {
                "question_id": qid,
                "refs": refs,
                "pred_256": p256,
                "pred_500": p500,
                "anls_256": s256,
                "anls_500": s500,
                "delta_anls": delta,
                "em_256": e256,
                "em_500": e500,
                "em_diag_256": d256,
                "em_diag_500": d500,
                "endswith_dot_256": ends_with_period_raw(p256),
                "endswith_dot_500": ends_with_period_raw(p500),
            }
        )

    mean_anls_256 = sum(r["anls_256"] for r in rows) / 208
    mean_anls_500 = sum(r["anls_500"] for r in rows) / 208
    mean_em_256 = sum(r["em_256"] for r in rows) / 208
    mean_em_500 = sum(r["em_500"] for r in rows) / 208
    # Net ANLS mean delta explained by sum_pos + sum_neg
    net_from_sums = (sum_pos + sum_neg) / 208

    # Confirm prior reported aggregates
    confirm = {
        "mean_anls_256_matches_prior": abs(mean_anls_256 - prev["smolvlm_256m"]["anls_strict_v2"])
        < 1e-9,
        "mean_anls_500_matches_prior": abs(mean_anls_500 - prev["smolvlm_500m"]["anls_strict_v2"])
        < 1e-9,
        "mean_em_256_matches_prior": abs(mean_em_256 - prev["smolvlm_256m"]["exact_match"])
        < 1e-9,
        "mean_em_500_matches_prior": abs(mean_em_500 - prev["smolvlm_500m"]["exact_match"])
        < 1e-9,
        "n_improved_matches_prior": n_imp
        == prev["per_question"]["n_improved_500_vs_256"],
        "n_regressed_matches_prior": n_reg
        == prev["per_question"]["n_regressed_500_vs_256"],
        "n_tied_matches_prior": n_tie == prev["per_question"]["n_tied"],
        "recomputed_mean_anls_256": mean_anls_256,
        "recomputed_mean_anls_500": mean_anls_500,
        "recomputed_mean_em_256": mean_em_256,
        "recomputed_mean_em_500": mean_em_500,
        "n_improved": n_imp,
        "n_regressed": n_reg,
        "n_tied": n_tie,
        "sum_positive_anls_deltas": sum_pos,
        "sum_negative_anls_deltas": sum_neg,
        "net_mean_delta_from_sums": net_from_sums,
        "observed_mean_delta": mean_anls_500 - mean_anls_256,
        "explanation": (
            "Higher 500M mean ANLS despite more regressions is explained by "
            "sum(|positive deltas|) exceeding sum(|negative deltas|): "
            f"sum_pos={sum_pos:.6f}, sum_neg={sum_neg:.6f}, "
            f"net/208={net_from_sums:.6f}."
        ),
    }
    assert all(
        confirm[k]
        for k in (
            "mean_anls_256_matches_prior",
            "mean_anls_500_matches_prior",
            "mean_em_256_matches_prior",
            "mean_em_500_matches_prior",
            "n_improved_matches_prior",
            "n_regressed_matches_prior",
            "n_tied_matches_prior",
        )
    ), confirm
    assert abs(net_from_sums - (mean_anls_500 - mean_anls_256)) < 1e-12

    # --- Formatting diagnostic ---
    def fmt_stats(model: str, em_key: str, diag_key: str, dot_key: str) -> dict:
        em_n = sum(1 for r in rows if r[em_key] == 1.0)
        diag_n = sum(1 for r in rows if r[diag_key] == 1.0)
        newly = [
            r
            for r in rows
            if r[em_key] == 0.0 and r[diag_key] == 1.0
        ]
        still = [
            r
            for r in rows
            if r[diag_key] == 0.0
        ]
        dots = sum(1 for r in rows if r[dot_key])
        # Deterministic samples: sort by qid
        newly_s = sorted(newly, key=lambda r: r["question_id"])
        still_s = sorted(still, key=lambda r: r["question_id"])
        sample_new = [
            {
                "question_id": r["question_id"],
                "prediction": r[f"pred_{model}"],
                "refs": r["refs"],
                "primary_em": r[em_key],
                "diagnostic_em": r[diag_key],
            }
            for r in newly_s[:8]
        ]
        sample_still = [
            {
                "question_id": r["question_id"],
                "prediction": r[f"pred_{model}"],
                "refs": r["refs"],
                "primary_em": r[em_key],
                "diagnostic_em": r[diag_key],
                "endswith_dot": r[dot_key],
            }
            for r in still_s[:8]
        ]
        return {
            "model": model,
            "n_requested": 208,
            "primary_em_count": em_n,
            "primary_em_rate": em_n / 208,
            "diagnostic_em_count": diag_n,
            "diagnostic_em_rate": diag_n / 208,
            "newly_matched_count": len(newly),
            "remaining_unmatched_count": len(still),
            "outputs_ending_with_period": dots,
            "sample_newly_matched": sample_new,
            "sample_still_unmatched": sample_still,
            "limitation": (
                "Diagnostic only: after standard normalize, remove at most one "
                "terminal ASCII '.' from prediction and each reference. Does not "
                "remove internal punctuation, decimals, %, units, articles, or "
                "arbitrary suffixes. Terminal punctuation can be meaningful; this "
                "is not official EM and does not replace primary metrics."
            ),
        }

    fmt_256 = fmt_stats("256", "em_256", "em_diag_256", "endswith_dot_256")
    fmt_500 = fmt_stats("500", "em_500", "em_diag_500", "endswith_dot_500")

    # Does punctuation explain the entire EM gap?
    # Gap primary: em256 - em500. After diagnostic: diag256 - diag500.
    gap_primary = fmt_256["primary_em_count"] - fmt_500["primary_em_count"]
    gap_diag = fmt_256["diagnostic_em_count"] - fmt_500["diagnostic_em_count"]
    punctuation_claim = {
        "primary_em_count_gap_256_minus_500": gap_primary,
        "diagnostic_em_count_gap_256_minus_500": gap_diag,
        "500M_newly_matched_by_terminal_period_diag": fmt_500["newly_matched_count"],
        "256M_newly_matched_by_terminal_period_diag": fmt_256["newly_matched_count"],
        "punctuation_explains_entire_gap": gap_diag == 0
        and fmt_500["newly_matched_count"] >= gap_primary,
        "assessment": (
            "Punctuation does NOT explain the entire primary EM gap unless "
            "diagnostic gap is ~0 and newly-matched 500M covers the primary gap. "
            f"Primary gap={gap_primary}, diagnostic gap={gap_diag}, "
            f"500M newly matched={fmt_500['newly_matched_count']}."
        ),
    }

    # --- Output behavior ---
    def output_behavior(label: str, by: dict[int, dict]) -> dict:
        lens: list[int] = []
        n_cap_true = 0
        n_cap_false = 0
        n_cap_missing = 0
        n_len_missing = 0
        long_ex = []
        for qid in qids:
            rec = by[qid]
            gl = rec.get("generated_token_length")
            if gl is None:
                n_len_missing += 1
            else:
                lens.append(int(gl))
            cap = rec.get("generation_cap_reached")
            if cap is None:
                # Historical 256M may lack the boolean; infer only if both lengths present
                if (
                    rec.get("generated_token_length") is not None
                    and rec.get("max_new_tokens") is not None
                ):
                    # Field not recorded as boolean — report unavailable for the flag
                    n_cap_missing += 1
                else:
                    n_cap_missing += 1
            elif cap:
                n_cap_true += 1
            else:
                n_cap_false += 1
            pred = rec.get("prediction") or ""
            if len(pred) >= 80 or (gl is not None and int(gl) >= 40):
                long_ex.append(
                    {
                        "question_id": qid,
                        "prediction": pred,
                        "generated_token_length": gl,
                        "char_len": len(pred),
                    }
                )
        long_ex = sorted(long_ex, key=lambda x: (-x["char_len"], x["question_id"]))[:6]

        # Literal reference-substring among EM failures (diagnostic only)
        substr_hits = []
        for qid in qids:
            rec = by[qid]
            pred = rec.get("prediction") or ""
            refs = ans_by[qid]
            if exact_match_single(pred, refs) == 1.0:
                continue
            pred_n = normalize_text(pred)
            hit_refs = []
            for ref in refs:
                rn = normalize_text(ref)
                if not rn:
                    continue
                if rn in pred_n or pred_n in rn:
                    hit_refs.append(ref)
            if hit_refs:
                substr_hits.append(
                    {
                        "question_id": qid,
                        "prediction": pred,
                        "matched_ref_substrings": hit_refs,
                    }
                )
        substr_hits = sorted(substr_hits, key=lambda x: x["question_id"])
        return {
            "model": label,
            "generated_token_length": {
                "n_available": len(lens),
                "n_unavailable": n_len_missing,
                "min": min(lens) if lens else None,
                "median": statistics.median(lens) if lens else None,
                "p95": percentile([float(x) for x in lens], 95),
                "max": max(lens) if lens else None,
                "mean": (sum(lens) / len(lens)) if lens else None,
            },
            "generation_cap_reached": {
                "n_true": n_cap_true,
                "n_false": n_cap_false,
                "n_unavailable": n_cap_missing,
                "note": (
                    "Missing historical boolean fields reported as unavailable, "
                    "not zero."
                ),
            },
            "long_or_repetitive_examples": long_ex,
            "em_failure_literal_ref_substring": {
                "n": len(substr_hits),
                "sample": substr_hits[:8],
                "caveat": (
                    "Diagnostic only. Short answers can match accidentally; "
                    "substring presence does not establish correctness."
                ),
            },
            "behavior_note": (
                "Observed token-length and punctuation patterns only; no proven "
                "OCR/reasoning cause assigned. No answer extraction or generation "
                "tuning introduced."
            ),
        }

    # For 256M, generation_cap_reached was often absent — check first record
    out_256 = output_behavior("256M", m256)
    out_500 = output_behavior("500M", m500)
    # Clarify 256M: if field absent on all, mark unavailable
    if all(m256[q].get("generation_cap_reached") is None for q in qids):
        # Optionally compute inferred cap from lengths for info, but keep reported unavailable
        inferred = sum(
            1
            for q in qids
            if m256[q].get("generated_token_length") is not None
            and int(m256[q]["generated_token_length"])
            >= int(m256[q].get("max_new_tokens") or 64)
        )
        out_256["generation_cap_reached"] = {
            "n_true": "unavailable",
            "n_false": "unavailable",
            "n_unavailable": 208,
            "inferred_from_lengths_ge_max_new_tokens": inferred,
            "note": (
                "Phase 2A JSONL did not store generation_cap_reached boolean; "
                "reported unavailable (not zero). Inferred count from lengths is "
                "informational only."
            ),
        }

    # --- Gallery provenance ---
    # Load pinned dataset rows for gallery qids (dev + smoke)
    from huggingface_hub import hf_hub_download, list_repo_files
    import pyarrow.parquet as pq

    need_gallery = set(GALLERY_QIDS)
    pinned_rows: dict[int, dict] = {}
    files = list_repo_files(DATASET_ID, repo_type="dataset", revision=DATASET_REV)
    val = sorted(
        f for f in files if f.startswith("data/validation-") and f.endswith(".parquet")
    )
    for shard in val:
        if len(pinned_rows) >= len(need_gallery):
            break
        path = hf_hub_download(
            DATASET_ID,
            filename=shard,
            repo_type="dataset",
            revision=DATASET_REV,
            cache_dir=str(Path(".hf_cache/hub")),
        )
        df = pq.read_table(
            path,
            columns=[
                "questionId",
                "docId",
                "ucsf_document_id",
                "ucsf_document_page_no",
                "question",
                "answers",
                "image",
            ],
        ).to_pandas()
        for _, r in df.iterrows():
            qid = int(r["questionId"])
            if qid not in need_gallery:
                continue
            img = r["image"]
            ib = bytes(img["bytes"]) if isinstance(img, dict) else bytes(img.get("bytes"))
            ucsf = r.get("ucsf_document_id")
            ucsf_s = None if ucsf is None else str(ucsf).strip() or None
            page = r.get("ucsf_document_page_no")
            page_s = None if page is None else str(page).strip() or None
            pinned_rows[qid] = {
                "question_id": qid,
                "doc_id": int(r["docId"]),
                "ucsf_document_id": ucsf_s,
                "ucsf_document_page_no": page_s,
                "question": str(r["question"]),
                "answers": [str(a) for a in list(r["answers"])],
                "source_image_sha256": sha256_bytes(ib),
            }

    assert set(pinned_rows) == need_gallery

    gallery_tables = []
    impact = {
        "wrong_ids_in_prior_gallery_docs": GALLERY_DOC_ERRORS,
        "manifests_matched_pinned_source": True,
        "audit_inputs_used_manifest_or_parquet_fields": True,
        "scope": (
            "Earlier wrong UCSF/doc_id for qids 5158 and 5263 were confined to "
            "gallery documentation (docs/examples). Development/smoke manifests "
            "and leakage-audit inputs already carried the correct UCSF/doc_id/"
            "page from the pinned dataset. No split membership change required "
            "for this documentation bug."
        ),
    }

    for qid in GALLERY_QIDS:
        pinned = pinned_rows[qid]
        if qid in man_by:
            man_e = man_by[qid]
            split = "development"
            manifest_name = "docvqa_dev_v1"
        else:
            man_e = smoke_by[qid]
            split = "smoke"
            manifest_name = "docvqa_smoke_v1"
        img_path = Path(man_e["image_relpath"])
        assert img_path.is_file(), f"missing image {img_path}"
        png_sha = sha256_file(img_path)
        im = Image.open(img_path).convert("RGB")
        w, h = im.size
        # Manifest vs pinned
        assert int(man_e["doc_id"]) == pinned["doc_id"], (qid, man_e, pinned)
        assert str(man_e.get("ucsf_document_id") or "") == str(
            pinned["ucsf_document_id"] or ""
        )
        assert str(man_e.get("ucsf_document_page_no") or "") == str(
            pinned["ucsf_document_page_no"] or ""
        )
        assert man_e["question"] == pinned["question"]
        if man_e.get("source_image_sha256"):
            assert man_e["source_image_sha256"] == pinned["source_image_sha256"]
        if man_e.get("image_sha256_png"):
            assert man_e["image_sha256_png"] == png_sha

        doc_err = None
        if qid in GALLERY_DOC_ERRORS:
            wrong = GALLERY_DOC_ERRORS[qid]
            doc_err = {
                "prior_gallery_doc_id": wrong["doc_id"],
                "prior_gallery_ucsf": wrong["ucsf_document_id"],
                "prior_gallery_page": wrong["ucsf_document_page_no"],
                "correct_doc_id": pinned["doc_id"],
                "correct_ucsf": pinned["ucsf_document_id"],
                "correct_page": pinned["ucsf_document_page_no"],
                "affected": "gallery_documentation_only",
            }

        pred_256 = None
        pred_500 = None
        scores = None
        if split == "development":
            pred_256 = m256[qid]["prediction"]
            pred_500 = m500[qid]["prediction"]
            refs = ans_by[qid]
            assert refs == pinned["answers"] or set(refs) == set(pinned["answers"])
            scores = {
                "anls_strict_v2_256": anls_normalized_strict_v2(pred_256, refs),
                "anls_strict_v2_500": anls_normalized_strict_v2(pred_500, refs),
                "em_256": exact_match_single(pred_256, refs),
                "em_500": exact_match_single(pred_500, refs),
            }
        else:
            # smoke: load smoke prediction if present
            smoke_pred_path = Path("results/smoke_smolvlm256m.jsonl")
            smoke_preds = load_last_by_qid(smoke_pred_path) if smoke_pred_path.is_file() else {}
            pred_256 = (smoke_preds.get(qid) or {}).get("prediction")
            pred_500 = None  # intentionally not invented

        gallery_tables.append(
            {
                "question_id": qid,
                "split": split,
                "manifest": manifest_name,
                "doc_id": pinned["doc_id"],
                "ucsf_document_id": pinned["ucsf_document_id"],
                "ucsf_document_page_no": pinned["ucsf_document_page_no"],
                "question": pinned["question"],
                "references": pinned["answers"],
                "image_relpath": man_e["image_relpath"],
                "image_width": w,
                "image_height": h,
                "image_sha256_png": png_sha,
                "source_image_sha256": pinned["source_image_sha256"],
                "prediction_256": pred_256,
                "prediction_500": pred_500,
                "scores": scores,
                "documentation_correction": doc_err,
                "provenance_ok": True,
            }
        )

    # Generate markdown tables programmatically
    md_lines = [
        "# Gallery tables (programmatically generated)",
        "",
        "**Source:** pinned `HuggingFaceM4/DocumentVQA` @ "
        f"`{DATASET_REV}` + frozen manifests + saved predictions.",
        "",
        "## Correction note (history preserved)",
        "",
        "Phase 2A gallery docs incorrectly listed qid **5158** as doc_id 1565 / "
        "UCSF `yjbj0227` p.1 and qid **5263** as doc_id 1585 / UCSF `yjbj0227` p.21. "
        "Authoritative manifests and the pinned dataset record **5158** → doc_id "
        "**1762** / `gzyh0227` p.**9** and **5263** → doc_id **1785** / `mtnh0227` "
        "p.**10**. Impact: **gallery documentation only**; manifests and leakage "
        "audits were already correct and were not rewritten for this bug.",
        "",
        "Primary scores use `anls_normalized_strict_v2`. Smoke qid 57368 has no 500M prediction.",
        "",
    ]
    for g in gallery_tables:
        md_lines.append(f"## qid `{g['question_id']}` ({g['split']})")
        md_lines.append("")
        md_lines.append("| Field | Value |")
        md_lines.append("|-------|-------|")
        md_lines.append(f"| question_id | {g['question_id']} |")
        md_lines.append(f"| doc_id | {g['doc_id']} |")
        md_lines.append(f"| ucsf_document_id | `{g['ucsf_document_id']}` |")
        md_lines.append(f"| page | {g['ucsf_document_page_no']} |")
        md_lines.append(f"| image | {g['image_width']}×{g['image_height']} |")
        md_lines.append(f"| image_sha256_png | `{g['image_sha256_png'][:16]}…` |")
        md_lines.append(f"| source_image_sha256 | `{g['source_image_sha256'][:16]}…` |")
        md_lines.append(f"| question | {g['question']} |")
        md_lines.append(f"| references | {g['references']} |")
        md_lines.append(f"| prediction_256 | {g['prediction_256']!r} |")
        if g["split"] == "development":
            md_lines.append(f"| prediction_500 | {g['prediction_500']!r} |")
            sc = g["scores"]
            md_lines.append(
                f"| ANLS strict / EM 256 | {sc['anls_strict_v2_256']:.4f} / {sc['em_256']:.1f} |"
            )
            md_lines.append(
                f"| ANLS strict / EM 500 | {sc['anls_strict_v2_500']:.4f} / {sc['em_500']:.1f} |"
            )
        else:
            md_lines.append("| prediction_500 | not evaluated (smoke-only) |")
        if g["documentation_correction"]:
            c = g["documentation_correction"]
            md_lines.append(
                f"| prior gallery error | doc {c['prior_gallery_doc_id']} / "
                f"`{c['prior_gallery_ucsf']}` p.{c['prior_gallery_page']} → "
                f"corrected to {c['correct_doc_id']} / `{c['correct_ucsf']}` "
                f"p.{c['correct_page']} (docs only) |"
            )
        md_lines.append("")

    Path("docs/examples/gallery_tables_generated.md").write_text(
        "\n".join(md_lines) + "\n"
    )

    report = {
        "task": "phase02b_cpu_followup_analysis",
        "n_requested": 208,
        "validation": confirm,
        "formatting_diagnostic": {
            "smolvlm_256m": fmt_256,
            "smolvlm_500m": fmt_500,
            "gap_assessment": punctuation_claim,
        },
        "output_behavior": {
            "smolvlm_256m": out_256,
            "smolvlm_500m": out_500,
        },
        "gallery_provenance": {
            "impact": impact,
            "examples": gallery_tables,
            "generated_markdown": "docs/examples/gallery_tables_generated.md",
        },
    }
    out_path = Path("results/phase02b_cpu_followup_analysis.json")
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "ok": True,
        "anls_256": mean_anls_256,
        "anls_500": mean_anls_500,
        "sum_pos": sum_pos,
        "sum_neg": sum_neg,
        "em_256": fmt_256["primary_em_count"],
        "em_diag_256": fmt_256["diagnostic_em_count"],
        "em_500": fmt_500["primary_em_count"],
        "em_diag_500": fmt_500["diagnostic_em_count"],
        "newly_500": fmt_500["newly_matched_count"],
        "dots_256": fmt_256["outputs_ending_with_period"],
        "dots_500": fmt_500["outputs_ending_with_period"],
        "punct_explains_entire_gap": punctuation_claim["punctuation_explains_entire_gap"],
        "gallery_n": len(gallery_tables),
    }, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
