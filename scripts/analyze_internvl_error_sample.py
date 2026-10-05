#!/usr/bin/env python3
"""Preliminary descriptive error sample from verified InternVL development predictions.

Selects up to 20 non-perfect examples with a fixed seed. Does not change metrics
or answers. Image-inspection labels are filled when images can be opened; uncertain
cases are marked explicitly. String mismatch alone does not imply visual failure.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def load_answers(path: Path) -> dict[int, list]:
    obj = json.loads(path.read_text())
    if isinstance(obj, dict) and "answers" in obj:
        return {int(r["question_id"]): list(r["answers"]) for r in obj["answers"]}
    if isinstance(obj, dict) and "records" in obj:
        return {int(r["question_id"]): list(r["answers"]) for r in obj["records"]}
    if isinstance(obj, list):
        return {int(r["question_id"]): list(r["answers"]) for r in obj}
    raise ValueError(f"Unrecognized answers sidecar format: {path}")


def metric_fields(row: dict) -> tuple[float | None, float | None]:
    m = row.get("metrics") or {}
    anls = m.get("anls_normalized_strict_v2")
    if anls is None:
        anls = row.get("anls_normalized_strict_v2")
    em = m.get("exact_match")
    if em is None:
        em = row.get("exact_match")
    return (
        None if anls is None else float(anls),
        None if em is None else float(em),
    )


def classify_with_image(
    *,
    prediction: str,
    references: list,
    image_path: Path | None,
) -> tuple[list[str], str, bool]:
    """Return (tags, uncertainty_note, image_inspected)."""
    tags: list[str] = []
    pred = (prediction or "").strip()
    refs = [str(a).strip() for a in references]
    pred_l = pred.lower()
    refs_l = [r.lower() for r in refs]

    image_inspected = False
    observed_text_hint = ""
    if image_path is not None and image_path.is_file():
        try:
            from PIL import Image

            im = Image.open(image_path).convert("RGB")
            image_inspected = True
            observed_text_hint = f"image_opened size={im.size[0]}x{im.size[1]}"
        except Exception as exc:  # noqa: BLE001
            observed_text_hint = f"image_open_failed: {exc}"

    # Formatting-only heuristics (still uncertain without proving OCR match).
    if any(pred_l == r + "." or r == pred_l + "." for r in refs_l):
        tags.append("formatting_difference")
    elif any(
        pred_l.replace(" ", "") == r.replace(" ", "") for r in refs_l if r and pred_l
    ):
        tags.append("formatting_difference")
    elif any(pred_l in r or r in pred_l for r in refs_l if r and pred_l):
        tags.append("possible_partial_span_or_entity_uncertain")
    else:
        tags.append("content_mismatch_string_level")

    if image_inspected:
        # Without OCR/layout tooling, do not assert cell/field selection from pixels.
        tags.append("image_opened_no_ocr_layout_tooling")
        uncertainty = (
            f"{observed_text_hint}. Opened image for presence only; "
            "text-reading vs wrong-field vs ambiguous-annotation not asserted "
            "without readable OCR/layout evidence. String mismatch alone is "
            "insufficient to claim visual reasoning failure."
        )
    else:
        tags.append("image_not_inspected")
        uncertainty = (
            "Image not available for inspection in this pass. "
            "Do not infer visual reasoning failure from string mismatch alone."
        )

    tags.append("descriptive_only")
    return tags, uncertainty, image_inspected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="results/dev_internvl3_1b.jsonl")
    parser.add_argument("--answers", default="data/cache/docvqa_dev_v1_answers.json")
    parser.add_argument("--manifest", default="data/manifests/docvqa_dev_v1.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-n", type=int, default=20)
    parser.add_argument("--out-json", default="results/internvl_error_sample_v1.json")
    parser.add_argument("--out-md", default="docs/progress/phase02d_error_sample.md")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    preds = [
        json.loads(line)
        for line in Path(args.predictions).read_text().splitlines()
        if line.strip()
    ]
    answers = load_answers(Path(args.answers))
    man = json.loads(Path(args.manifest).read_text())
    by_qid = {int(e["question_id"]): e for e in man["examples"]}

    non_perfect = []
    for p in preds:
        if p.get("status") != "ok":
            continue
        qid = int(p["question_id"])
        anls, em = metric_fields(p)
        is_perfect = (
            em is not None
            and em >= 1.0 - 1e-12
            and (anls is None or anls >= 1.0 - 1e-12)
        )
        if is_perfect:
            continue
        ex = by_qid.get(qid, {})
        refs = answers.get(qid) or p.get("reference_answers") or []
        non_perfect.append(
            {
                "question_id": qid,
                "question": ex.get("question") or p.get("question"),
                "prediction": p.get("prediction"),
                "references": list(refs),
                "image_relpath": ex.get("image_relpath") or p.get("image_relpath"),
                "anls_normalized_strict_v2": anls,
                "exact_match": em,
                "ucsf_document_id": ex.get("ucsf_document_id"),
                "ucsf_document_page_no": ex.get("ucsf_document_page_no"),
            }
        )

    non_perfect.sort(key=lambda r: int(r["question_id"]))
    rng = random.Random(int(args.seed))
    pool = list(non_perfect)
    rng.shuffle(pool)
    sample = pool[: int(args.max_n)]
    sample.sort(key=lambda r: int(r["question_id"]))

    n_img = 0
    for row in sample:
        img = root / row["image_relpath"] if row.get("image_relpath") else None
        tags, uncertainty, inspected = classify_with_image(
            prediction=row["prediction"] or "",
            references=row["references"],
            image_path=img,
        )
        row["image_file_present"] = bool(img and img.is_file())
        row["image_inspected"] = inspected
        row["error_tags"] = tags
        row["uncertainty"] = uncertainty
        if inspected:
            n_img += 1

    report = {
        "task": "internvl_error_sample_v1",
        "seed": int(args.seed),
        "max_n": int(args.max_n),
        "n_predictions": len(preds),
        "n_non_perfect": len(non_perfect),
        "n_sampled": len(sample),
        "n_images_opened": n_img,
        "selection": (
            f"status=ok InternVL development rows that are not perfect "
            f"(exact_match and ANLS≈1); sort by qid; shuffle with "
            f"random.Random({args.seed}); take first {args.max_n}; re-sort by qid."
        ),
        "predictions_path": args.predictions,
        "note": (
            "Descriptive only. Do not change benchmark answers or metrics. "
            "Do not infer visual reasoning failure solely from string mismatch."
        ),
        "examples": sample,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# Phase 2D — preliminary InternVL error sample",
        "",
        f"**Seed:** {args.seed}  ",
        f"**Sampled:** {len(sample)} / {len(non_perfect)} non-perfect of {len(preds)} development predictions  ",
        f"**Images opened:** {n_img}/{len(sample)}  ",
        f"**Source predictions:** `{args.predictions}`  ",
        "",
        "Descriptive only. Tags ending in `_uncertain` are not confirmed. "
        "String mismatch alone does **not** imply a visual reasoning failure.",
        "",
    ]
    for i, row in enumerate(sample, 1):
        lines.append(f"## {i}. qid {row['question_id']}")
        lines.append("")
        lines.append(
            f"- **Image:** `{row['image_relpath']}` "
            f"(present={row['image_file_present']}, inspected={row['image_inspected']})"
        )
        lines.append(
            f"- **UCSF:** `{row.get('ucsf_document_id')}` page "
            f"`{row.get('ucsf_document_page_no')}`"
        )
        lines.append(
            f"- **ANLS strict v2:** {row['anls_normalized_strict_v2']}  "
            f"**EM:** {row['exact_match']}"
        )
        lines.append(f"- **Tags:** {', '.join(row['error_tags'])}")
        lines.append(f"- **Uncertainty:** {row['uncertainty']}")
        lines.append("")
        lines.append(f"**Q:** {row['question']}")
        lines.append("")
        lines.append(f"**Prediction:** {row['prediction']!r}")
        lines.append("")
        lines.append(f"**References:** {row['references']}")
        lines.append("")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps({"out_json": str(out_json), "out_md": str(out_md), "n": len(sample)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
