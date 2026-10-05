#!/usr/bin/env python3
"""Deterministically rebuild holdout v2 by UCSF source-document groups.

Keeps fixed development (and smoke) question IDs. Seed 42. Target ~500 questions.
Independent of model predictions.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image

DATASET_ID = "HuggingFaceM4/DocumentVQA"
DATASET_REV = "a44195f8f989c10d06ab4ccc98ffe9cdb6d928e4"
SEED = 42
TARGET = 500


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


def main() -> int:
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        print("ERROR: PYTHONNOUSERSITE=1 required", file=sys.stderr)
        return 2

    root = Path(__file__).resolve().parents[1]
    os.chdir(root)

    smoke = json.loads(Path("data/manifests/docvqa_smoke_v1.json").read_text())
    dev = json.loads(Path("data/manifests/docvqa_dev_v1.json").read_text())

    hist = Path("data/manifests/historical")
    hist.mkdir(parents=True, exist_ok=True)
    if not (hist / "docvqa_holdout_v1.superseded.json").exists():
        shutil.copy(
            "data/manifests/docvqa_holdout_v1.json",
            hist / "docvqa_holdout_v1.superseded.json",
        )
        print("preserved historical holdout_v1")

    blocked_ucsf: set[str] = set()
    blocked_qids: set[int] = set()
    for e in smoke["examples"] + dev["examples"]:
        blocked_qids.add(int(e["question_id"]))
        u = str(e.get("ucsf_document_id") or "").strip()
        if u:
            blocked_ucsf.add(u)
    print("blocked_ucsf", len(blocked_ucsf), "blocked_qids", len(blocked_qids))

    files = list_repo_files(DATASET_ID, repo_type="dataset", revision=DATASET_REV)
    val = sorted(
        f for f in files if f.startswith("data/validation-") and f.endswith(".parquet")
    )
    by_group: dict[tuple, list] = defaultdict(list)
    all_rows: list = []
    for shard in val:
        path = hf_hub_download(
            DATASET_ID,
            filename=shard,
            repo_type="dataset",
            revision=DATASET_REV,
            cache_dir=str(Path(".hf_cache/hub")),
        )
        df = pq.read_table(path).to_pandas()
        for _, r in df.iterrows():
            qid = int(r["questionId"])
            ucsf = r.get("ucsf_document_id")
            ucsf_s = None if ucsf is None else str(ucsf).strip() or None
            page = r.get("ucsf_document_page_no")
            page_s = None if page is None else str(page).strip() or None
            img = r["image"]
            ib = bytes(img["bytes"]) if isinstance(img, dict) else bytes(img.get("bytes"))
            row = {
                "question_id": qid,
                "doc_id": int(r["docId"]),
                "ucsf": ucsf_s,
                "page": page_s,
                "question": str(r["question"]),
                "answers": [str(a) for a in list(r["answers"])],
                "_image_bytes": ib,
                "source_image_sha256": sha256_bytes(ib),
            }
            all_rows.append(row)
            if ucsf_s is None:
                key = ("PAGE_FALLBACK", row["doc_id"], page_s, row["source_image_sha256"])
            else:
                key = ("UCSF", ucsf_s)
            by_group[key].append(row)
        print("loaded", shard, "rows", len(all_rows))

    eligible_keys = []
    for key, rows in by_group.items():
        if key[0] == "UCSF" and key[1] in blocked_ucsf:
            continue
        if any(int(r["question_id"]) in blocked_qids for r in rows):
            continue
        eligible_keys.append(key)

    rng = random.Random(SEED)
    rng.shuffle(eligible_keys)

    selected_keys: list = []
    total_q = 0
    for key in eligible_keys:
        n = len(by_group[key])
        if total_q >= TARGET:
            break
        selected_keys.append(key)
        total_q += n
    while len(selected_keys) > 1:
        last = selected_keys[-1]
        if total_q - len(by_group[last]) >= TARGET:
            selected_keys.pop()
            total_q -= len(by_group[last])
        else:
            break

    hold_rows = []
    for key in sorted(selected_keys, key=lambda k: (k[0],) + tuple(str(x) for x in k[1:])):
        hold_rows.extend(sorted(by_group[key], key=lambda r: int(r["question_id"])))

    print("holdout_v2 n", len(hold_rows), "groups", len(selected_keys))

    hold_ucsf = {r["ucsf"] for r in hold_rows if r["ucsf"]}
    overlap = hold_ucsf & blocked_ucsf
    if overlap:
        raise SystemExit(f"FAIL leakage UCSF overlap: {sorted(overlap)}")
    if set(int(r["question_id"]) for r in hold_rows) & blocked_qids:
        raise SystemExit("FAIL leakage qid overlap")

    img_dir = Path("data/images/docvqa_holdout_v2")
    img_dir.mkdir(parents=True, exist_ok=True)
    examples = []
    answers = []
    for rank, r in enumerate(hold_rows):
        qid = int(r["question_id"])
        rel = f"data/images/docvqa_holdout_v2/qid_{qid}.png"
        Image.open(io.BytesIO(r["_image_bytes"])).convert("RGB").save(rel, format="PNG")
        png_hash = sha256_bytes(Path(rel).read_bytes())
        examples.append(
            {
                "index": rank,
                "question_id": qid,
                "doc_id": int(r["doc_id"]),
                "ucsf_document_id": r["ucsf"],
                "ucsf_document_page_no": r["page"],
                "grouping_key": "ucsf_document_id" if r["ucsf"] else "page_fallback",
                "question": r["question"],
                "image_relpath": rel,
                "image_sha256_png": png_hash,
                "source_image_sha256": r["source_image_sha256"],
                "n_answers": len(r["answers"]),
            }
        )
        answers.append({"question_id": qid, "answers": r["answers"]})

    body = {
        "name": "docvqa_holdout_v2",
        "role": "holdout",
        "supersedes": "docvqa_holdout_v1",
        "n_examples": len(examples),
        "n_docs_field_doc_id": len({e["doc_id"] for e in examples}),
        "n_source_groups_ucsf_or_fallback": len(selected_keys),
        "target_questions_approx": TARGET,
        "question_ids": [e["question_id"] for e in examples],
        "doc_ids": sorted({e["doc_id"] for e in examples}),
        "ucsf_document_ids": sorted(
            {e["ucsf_document_id"] for e in examples if e["ucsf_document_id"]}
        ),
        "metadata_distribution": {
            "n_questions": len(examples),
            "n_groups": len(selected_keys),
            "answers_per_q_mean": sum(e["n_answers"] for e in examples) / len(examples),
        },
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REV,
        "split": "validation",
        "selection_seed": SEED,
        "grouping_key": "ucsf_document_id",
        "grouping_note": (
            "Groups by UCSF source-document ID when present. Missing UCSF uses "
            "page-level fallback (PAGE_FALLBACK, doc_id, page, source_image_sha256); "
            "never empty-string collapse. doc_id alone is NOT a reliable whole-PDF "
            "grouping key (178 UCSF IDs span multiple doc_ids in validation)."
        ),
        "smoke_and_dev_exclusion": {
            "blocked_ucsf_document_ids": sorted(blocked_ucsf),
            "blocked_question_ids_n": len(blocked_qids),
        },
        "future_training_check_requirement": (
            "Any future training sample must be checked against development and "
            "holdout UCSF source identities and page/RGB/source hashes before use."
        ),
        "examples": examples,
    }
    body_for_hash = {k: v for k, v in body.items() if k != "manifest_sha256"}
    body["manifest_sha256"] = sha256_text(
        json.dumps(body_for_hash, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    )
    Path("data/manifests/docvqa_holdout_v2.json").write_text(
        json.dumps(body, indent=2) + "\n"
    )
    Path("data/cache/docvqa_holdout_v2_answers.json").write_text(
        json.dumps({"answers": answers}, indent=2) + "\n"
    )
    note = {
        "status": "superseded",
        "superseded_by": "docvqa_holdout_v2",
        "reason": (
            "UCSF source-document overlap with development (6 IDs) under corrected grouping"
        ),
        "historical_copy": "data/manifests/historical/docvqa_holdout_v1.superseded.json",
    }
    Path("data/manifests/docvqa_holdout_v1.SUPERSEDED.md").write_text(
        "# SUPERSEDED\n\n" + json.dumps(note, indent=2) + "\n"
    )
    print("wrote holdout_v2", body["n_examples"], "sha", body["manifest_sha256"][:16])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
