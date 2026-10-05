#!/usr/bin/env python3
"""Build frozen DocVQA development and final-holdout manifests.

Samples from the *full* HuggingFaceM4/DocumentVQA validation split (all shards),
group-disjoint by ``doc_id``, seed 42. Smoke documents/pages are excluded from
the final holdout. Images and answer sidecars are written under gitignored paths.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_image_bytes(img_obj) -> bytes:
    if isinstance(img_obj, dict) and img_obj.get("bytes") is not None:
        return bytes(img_obj["bytes"])
    if hasattr(img_obj, "get") and img_obj.get("bytes") is not None:
        return bytes(img_obj["bytes"])
    raise TypeError(f"Unsupported image payload type: {type(img_obj)}")


def load_smoke_exclusions(smoke_manifest: Path) -> tuple[set[int], set[tuple[str, str]], set[int]]:
    data = json.loads(smoke_manifest.read_text())
    doc_ids = set()
    pages = set()
    qids = set()
    for ex in data["examples"]:
        doc_ids.add(int(ex["doc_id"]))
        pages.add((str(ex.get("ucsf_document_id", "")), str(ex.get("ucsf_document_page_no", ""))))
        qids.add(int(ex["question_id"]))
    return doc_ids, pages, qids


def write_examples(
    examples: list[dict],
    *,
    image_dir: Path,
    image_prefix: str,
) -> tuple[list[dict], list[dict]]:
    image_dir.mkdir(parents=True, exist_ok=True)
    out_ex = []
    answers = []
    for rank, row in enumerate(examples):
        qid = int(row["question_id"])
        img_bytes = row["_image_bytes"]
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        rel = f"{image_prefix}/qid_{qid}.png"
        out_path = Path(rel)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path, format="PNG")
        png_hash = sha256_bytes(out_path.read_bytes())
        out_ex.append(
            {
                "index": rank,
                "question_id": qid,
                "doc_id": int(row["doc_id"]),
                "ucsf_document_id": str(row["ucsf_document_id"]),
                "ucsf_document_page_no": str(row["ucsf_document_page_no"]),
                "question": str(row["question"]),
                "image_relpath": rel,
                "image_sha256_png": png_hash,
                "source_image_sha256": sha256_bytes(img_bytes),
                "n_answers": len(row["answers"]),
            }
        )
        answers.append({"question_id": qid, "answers": list(row["answers"])})
    return out_ex, answers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default="HuggingFaceM4/DocumentVQA")
    parser.add_argument(
        "--dataset-revision",
        default="a44195f8f989c10d06ab4ccc98ffe9cdb6d928e4",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-target", type=int, default=200)
    parser.add_argument("--holdout-target", type=int, default=500)
    parser.add_argument(
        "--smoke-manifest",
        default="data/manifests/docvqa_smoke_v1.json",
    )
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HOME", ".hf_cache"))
    parser.add_argument("--dev-manifest", default="data/manifests/docvqa_dev_v1.json")
    parser.add_argument(
        "--holdout-manifest", default="data/manifests/docvqa_holdout_v1.json"
    )
    parser.add_argument(
        "--dev-answers", default="data/cache/docvqa_dev_v1_answers.json"
    )
    parser.add_argument(
        "--holdout-answers", default="data/cache/docvqa_holdout_v1_answers.json"
    )
    parser.add_argument("--inspect-train-only-meta", action="store_true")
    args = parser.parse_args()

    cache_root = Path(args.cache_dir).resolve()
    hub_cache = cache_root / "hub"
    hub_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)

    files = list_repo_files(
        args.dataset_id, repo_type="dataset", revision=args.dataset_revision
    )
    val_shards = sorted(f for f in files if f.startswith("data/validation-") and f.endswith(".parquet"))
    train_shards = sorted(f for f in files if f.startswith("data/train-") and f.endswith(".parquet"))
    print(f"validation_shards={len(val_shards)} train_shards={len(train_shards)}")
    if args.inspect_train_only_meta:
        print("train_shards_list_head=", train_shards[:3])
        print("NOTE: not downloading train corpus in Phase 2A")
        return 0

    smoke_docs, smoke_pages, smoke_qids = load_smoke_exclusions(Path(args.smoke_manifest))
    print(f"smoke_docs={sorted(smoke_docs)} smoke_qids={sorted(smoke_qids)}")

    rows: list[dict] = []
    for shard in val_shards:
        print(f"download {shard}")
        path = hf_hub_download(
            repo_id=args.dataset_id,
            repo_type="dataset",
            filename=shard,
            revision=args.dataset_revision,
            cache_dir=str(hub_cache),
        )
        table = pq.read_table(path)
        df = table.to_pandas()
        for _, r in df.iterrows():
            img_bytes = extract_image_bytes(r["image"])
            answers = [str(a) for a in list(r["answers"])]
            rows.append(
                {
                    "question_id": int(r["questionId"]),
                    "doc_id": int(r["docId"]),
                    "ucsf_document_id": str(r.get("ucsf_document_id", "")),
                    "ucsf_document_page_no": str(r.get("ucsf_document_page_no", "")),
                    "question": str(r["question"]),
                    "answers": answers,
                    "_image_bytes": img_bytes,
                    "_source_image_sha256": sha256_bytes(img_bytes),
                    "_shard": shard,
                }
            )
        print(f"  cumulative_rows={len(rows)}")

    # Group by document identity (doc_id). Limitation: pages of the same PDF
    # share doc_id; questions on different pages of one doc stay together.
    by_doc: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_doc[int(row["doc_id"])].append(row)

    all_doc_ids = sorted(by_doc.keys())
    rng = random.Random(args.seed)
    rng.shuffle(all_doc_ids)

    # Holdout first: exclude smoke docs entirely from holdout.
    holdout_docs: list[int] = []
    holdout_q = 0
    for did in all_doc_ids:
        if did in smoke_docs:
            continue
        # Also exclude if any smoke page identity appears (belt-and-suspenders)
        if any(
            (r["ucsf_document_id"], r["ucsf_document_page_no"]) in smoke_pages
            for r in by_doc[did]
        ):
            continue
        if holdout_q >= args.holdout_target:
            break
        holdout_docs.append(did)
        holdout_q += len(by_doc[did])

    holdout_doc_set = set(holdout_docs)
    rem_docs = [d for d in all_doc_ids if d not in holdout_doc_set]
    rng2 = random.Random(args.seed + 1)
    rng2.shuffle(rem_docs)

    # Dev may include smoke docs (they are only banned from final holdout),
    # but we still exclude smoke *question_ids* from both tracked splits to
    # avoid overlapping evaluation items with the smoke diagnostic set.
    # Prefer excluding smoke docs from both for cleaner separation.
    dev_docs: list[int] = []
    dev_q = 0
    for did in rem_docs:
        if did in smoke_docs:
            continue
        if dev_q >= args.dev_target:
            break
        dev_docs.append(did)
        dev_q += len(by_doc[did])

    def flatten(doc_list: list[int]) -> list[dict]:
        out = []
        for did in sorted(doc_list):  # stable order by doc_id after selection
            items = sorted(by_doc[did], key=lambda r: int(r["question_id"]))
            out.extend(items)
        return out

    holdout_rows = flatten(holdout_docs)
    # Trim excess questions from the last doc groups is NOT done — keep whole
    # documents so counts are approximate (>= target when last group overshoots).
    # If overshoot is large, drop trailing whole docs until >= target still holds
    # with minimal overshoot after removing one more would go below target.
    def trim_to_approx(doc_list: list[int], target: int) -> list[int]:
        # doc_list is in selection order; rebuild cumulative
        selected = []
        total = 0
        for did in doc_list:
            n = len(by_doc[did])
            if total >= target:
                break
            selected.append(did)
            total += n
        # Try dropping from the end while remaining >= target
        while len(selected) > 1:
            last = selected[-1]
            if total - len(by_doc[last]) >= target:
                selected.pop()
                total -= len(by_doc[last])
            else:
                break
        return selected

    holdout_docs = trim_to_approx(holdout_docs, args.holdout_target)
    dev_docs = trim_to_approx(dev_docs, args.dev_target)
    holdout_rows = flatten(holdout_docs)
    dev_rows = flatten(dev_docs)

    # Safety: disjoint docs and no smoke qids / smoke docs in holdout
    assert set(holdout_docs).isdisjoint(set(dev_docs))
    assert set(holdout_docs).isdisjoint(smoke_docs)
    assert smoke_qids.isdisjoint({int(r["question_id"]) for r in holdout_rows})
    assert smoke_qids.isdisjoint({int(r["question_id"]) for r in dev_rows})

    # Image-hash duplicate check across splits (page content)
    dev_hashes = {r["_source_image_sha256"] for r in dev_rows}
    hold_hashes = {r["_source_image_sha256"] for r in holdout_rows}
    hash_overlap = sorted(dev_hashes & hold_hashes)

    def meta_dist(example_rows: list[dict]) -> dict:
        n_ans = [len(r["answers"]) for r in example_rows]
        qlen = [len(r["question"]) for r in example_rows]
        return {
            "n_questions": len(example_rows),
            "n_docs": len({int(r["doc_id"]) for r in example_rows}),
            "answers_per_q_min": min(n_ans) if n_ans else None,
            "answers_per_q_max": max(n_ans) if n_ans else None,
            "answers_per_q_mean": (sum(n_ans) / len(n_ans)) if n_ans else None,
            "question_chars_min": min(qlen) if qlen else None,
            "question_chars_max": max(qlen) if qlen else None,
            "question_chars_mean": (sum(qlen) / len(qlen)) if qlen else None,
        }

    dev_out, dev_ans = write_examples(
        dev_rows, image_dir=Path("data/images/docvqa_dev"), image_prefix="data/images/docvqa_dev"
    )
    hold_out, hold_ans = write_examples(
        holdout_rows,
        image_dir=Path("data/images/docvqa_holdout"),
        image_prefix="data/images/docvqa_holdout",
    )

    common = {
        "dataset_id": args.dataset_id,
        "dataset_revision": args.dataset_revision,
        "split": "validation",
        "selection_seed": args.seed,
        "grouping_key": "doc_id",
        "grouping_note": (
            "Groups are DocVQA doc_id values (source document). All questions "
            "for a document stay in one split. This is stronger than page-level "
            "disjointness when a document has multiple pages."
        ),
        "smoke_exclusion": {
            "smoke_manifest": args.smoke_manifest,
            "excluded_doc_ids_from_holdout_and_dev": sorted(smoke_docs),
            "excluded_question_ids": sorted(smoke_qids),
        },
        "image_hash_overlap_dev_holdout": hash_overlap,
        "train_split_metadata_only": {
            "n_train_shards": len(train_shards),
            "note": "Full train corpus not downloaded in Phase 2A.",
        },
    }

    def dump_manifest(path: str, name: str, examples: list[dict], docs: list[int], rows: list[dict], target: int):
        body = {
            "name": name,
            "role": name.split("_")[1] if "_" in name else name,
            "n_examples": len(examples),
            "n_docs": len(docs),
            "target_questions_approx": target,
            "doc_ids": sorted(docs),
            "question_ids": [int(e["question_id"]) for e in examples],
            "metadata_distribution": meta_dist(rows),
            "examples": examples,
            **common,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(body, indent=2) + "\n"
        p.write_text(text)
        digest = hashlib.sha256(text.encode()).hexdigest()
        # Re-write with hash field
        body["manifest_sha256"] = digest
        # Stable: hash without the hash field
        body_for_hash = {k: v for k, v in body.items() if k != "manifest_sha256"}
        digest2 = hashlib.sha256(
            (json.dumps(body_for_hash, indent=2, sort_keys=True) + "\n").encode()
        ).hexdigest()
        body["manifest_sha256"] = digest2
        p.write_text(json.dumps(body, indent=2) + "\n")
        print(f"wrote {p} n={len(examples)} docs={len(docs)} sha256={digest2}")
        return digest2

    Path(args.dev_answers).parent.mkdir(parents=True, exist_ok=True)
    Path(args.dev_answers).write_text(json.dumps({"answers": dev_ans}, indent=2) + "\n")
    Path(args.holdout_answers).write_text(json.dumps({"answers": hold_ans}, indent=2) + "\n")

    dump_manifest(args.dev_manifest, "docvqa_dev_v1", dev_out, dev_docs, dev_rows, args.dev_target)
    dump_manifest(
        args.holdout_manifest, "docvqa_holdout_v1", hold_out, holdout_docs, holdout_rows, args.holdout_target
    )
    print(f"hash_overlap_count={len(hash_overlap)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
