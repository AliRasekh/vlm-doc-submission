#!/usr/bin/env python3
"""Freeze deterministic unrelated-image mapping (CPU; no answers in selection)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

from PIL import Image

from . import MAPPING_SEED, N_DEV
from .paths import MANIFEST, REPO_ROOT, VD_ART, assert_under_vd_artifacts


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _aspect_log(w: int, h: int) -> float:
    return math.log(max(w, 1) / max(h, 1))


def select_mapping_from_rows(
    rows: list[dict[str, Any]], *, seed: int = MAPPING_SEED
) -> dict[str, Any]:
    """Select unrelated donors from precomputed rows (CPU-pure; no image I/O).

    Each row must include: question_id, width, height, aspect_log (or width/height),
    image_relpath, and optionally doc_id / ucsf_* / image_sha256_png.
    """
    prepared: list[dict[str, Any]] = []
    for r in rows:
        row = dict(r)
        if "aspect_log" not in row:
            row["aspect_log"] = _aspect_log(int(row["width"]), int(row["height"]))
        prepared.append(row)

    by_qid = {int(r["question_id"]): r for r in prepared}
    rng = random.Random(seed)
    mapping: dict[str, Any] = {}
    for q in prepared:
        cands = []
        for d in prepared:
            if d["question_id"] == q["question_id"]:
                continue
            if q.get("doc_id") is not None and d.get("doc_id") == q.get("doc_id"):
                continue
            if (
                q.get("ucsf_document_id")
                and d.get("ucsf_document_id")
                and q.get("ucsf_document_id") == d.get("ucsf_document_id")
            ):
                continue
            if (
                q.get("ucsf_document_page_no") is not None
                and d.get("ucsf_document_page_no") is not None
                and q.get("ucsf_document_id")
                and d.get("ucsf_document_id")
                and q.get("ucsf_document_id") == d.get("ucsf_document_id")
                and str(q.get("ucsf_document_page_no")) == str(d.get("ucsf_document_page_no"))
            ):
                continue
            if q.get("image_sha256_png") and q.get("image_sha256_png") == d.get(
                "image_sha256_png"
            ):
                continue
            ar_dist = abs(q["aspect_log"] - d["aspect_log"])
            cands.append((ar_dist, d["question_id"], d))
        if not cands:
            raise RuntimeError(f"no unrelated donor for qid={q['question_id']}")
        cands.sort(key=lambda t: (t[0], t[1]))
        best_dist = cands[0][0]
        tied = [c for c in cands if abs(c[0] - best_dist) < 1e-15]
        if len(tied) == 1:
            chosen = tied[0][2]
            tie_break = "unique_min_aspect_distance"
        else:
            tied_ids = [c[1] for c in tied]
            pick = rng.choice(tied_ids)
            chosen = by_qid[pick]
            tie_break = "seeded_choice_among_equal_aspect_distance"
        mapping[str(q["question_id"])] = {
            "query_qid": q["question_id"],
            "donor_qid": chosen["question_id"],
            "query_image_relpath": q["image_relpath"],
            "donor_image_relpath": chosen["image_relpath"],
            "query_size_wh": [q["width"], q["height"]],
            "donor_size_wh": [chosen["width"], chosen["height"]],
            "aspect_log_distance": abs(q["aspect_log"] - chosen["aspect_log"]),
            "tie_break": tie_break,
            "query_doc_id": q.get("doc_id"),
            "donor_doc_id": chosen.get("doc_id"),
            "query_image_sha256_png": q.get("image_sha256_png"),
            "donor_image_sha256_png": chosen.get("image_sha256_png"),
            "resized_donor_to_query_geometry": False,
        }

    body = json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()
    mapping_sha = _sha256_bytes(body)
    return {
        "task": "pascal_visual_dependence_unrelated_mapping",
        "seed": seed,
        "n_queries": len(mapping),
        "manifest": str(MANIFEST.relative_to(REPO_ROOT)),
        "selection_rule": (
            "Exclude same doc_id / same ucsf_document_id / same image_sha256_png; "
            "prefer minimal |log(aspect)_query - log(aspect)_donor|; "
            "ties broken by seeded Random(seed).choice among equal-distance donors "
            "(candidates ordered by ascending donor qid before tie set). "
            "Answers and predictions are never used."
        ),
        "mapping_sha256": mapping_sha,
        "mapping": mapping,
    }


def build_mapping(seed: int = MAPPING_SEED) -> dict[str, Any]:
    man = json.loads(MANIFEST.read_text())
    examples = man.get("examples") or []
    if len(examples) != N_DEV:
        raise RuntimeError(f"expected {N_DEV} examples, got {len(examples)}")

    rows: list[dict[str, Any]] = []
    for ex in examples:
        img_path = REPO_ROOT / ex["image_relpath"]
        with Image.open(img_path) as im:
            w, h = im.size
        rows.append(
            {
                "question_id": int(ex["question_id"]),
                "doc_id": ex.get("doc_id"),
                "ucsf_document_id": ex.get("ucsf_document_id"),
                "ucsf_document_page_no": ex.get("ucsf_document_page_no"),
                "image_relpath": ex["image_relpath"],
                "image_sha256_png": ex.get("image_sha256_png"),
                "width": int(w),
                "height": int(h),
                "aspect_log": _aspect_log(int(w), int(h)),
            }
        )
    return select_mapping_from_rows(rows, seed=seed)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=MAPPING_SEED)
    parser.add_argument(
        "--out",
        default="companion/pascal/artifacts/visual_dep/unrelated_mapping.json",
    )
    args = parser.parse_args()
    out = assert_under_vd_artifacts(args.out, kind="mapping")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = build_mapping(args.seed)
    text = json.dumps(payload, indent=2) + "\n"
    out.write_text(text)
    # Full-file hash for inventory
    file_sha = _sha256_bytes(text.encode())
    meta = {
        "mapping_path": str(out.relative_to(REPO_ROOT)),
        "mapping_sha256": payload["mapping_sha256"],
        "file_sha256": file_sha,
        "seed": args.seed,
        "n": payload["n_queries"],
    }
    meta_path = out.with_name("unrelated_mapping_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
