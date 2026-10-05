#!/usr/bin/env python3
"""Build a leakage-checked DocVQA training subset (~2000 QA) from the pinned train split.

Deterministic (seed 42). Excludes UCSF / page / image hashes present in smoke,
development, or holdout v2. Does not use development predictions for selection.
Supervised target = first nonempty source-listed answer. All answers preserved
in the local sidecar. Data only — no training loop.

Performance: metadata-first scan (no image decode for UCSF-present rows); images
loaded only for PAGE_FALLBACK keys and for the final selected question IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image

DATASET_ID = "HuggingFaceM4/DocumentVQA"
DATASET_REV = "a44195f8f989c10d06ab4ccc98ffe9cdb6d928e4"
SEED = 42
TARGET_N = 2000
META_COLS = [
    "questionId",
    "docId",
    "ucsf_document_id",
    "ucsf_document_page_no",
    "question",
    "answers",
]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rgb_hash_from_bytes(img_bytes: bytes) -> str:
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return sha256_bytes(im.tobytes() + f"{im.size[0]}x{im.size[1]}".encode())


def rgb_hash_from_png(path: Path) -> str:
    im = Image.open(path).convert("RGB")
    return sha256_bytes(im.tobytes() + f"{im.size[0]}x{im.size[1]}".encode())


def first_valid_answer(answers: list) -> str | None:
    for a in answers:
        s = str(a).strip()
        if s:
            return s
    return None


def page_key(ucsf: str | None, page: str | None) -> tuple[str, str]:
    return (str(ucsf or ""), str(page or ""))


def load_eval_blocklists(root: Path) -> dict:
    manifests = {
        "smoke": root / "data/manifests/docvqa_smoke_v1.json",
        "dev": root / "data/manifests/docvqa_dev_v1.json",
        "holdout_v2": root / "data/manifests/docvqa_holdout_v2.json",
    }
    ucsf: set[str] = set()
    qids: set[int] = set()
    pages: set[tuple[str, str]] = set()
    src_hashes: set[str] = set()
    png_hashes: set[str] = set()
    rgb_hashes: set[str] = set()
    per_split: dict[str, dict] = {}

    for name, path in manifests.items():
        man = json.loads(path.read_text())
        s_ucsf: set[str] = set()
        s_qids: set[int] = set()
        s_pages: set[tuple[str, str]] = set()
        s_src: set[str] = set()
        s_png: set[str] = set()
        s_rgb: set[str] = set()
        for e in man["examples"]:
            qid = int(e["question_id"])
            s_qids.add(qid)
            qids.add(qid)
            u = str(e.get("ucsf_document_id") or "").strip()
            p = str(e.get("ucsf_document_page_no") or "").strip()
            if u:
                s_ucsf.add(u)
                ucsf.add(u)
            s_pages.add((u, p))
            pages.add((u, p))
            src = e.get("source_image_sha256")
            if src:
                s_src.add(src)
                src_hashes.add(src)
            png = e.get("image_sha256_png")
            if png:
                s_png.add(png)
                png_hashes.add(png)
            img_path = root / e["image_relpath"]
            if img_path.is_file():
                rh = rgb_hash_from_png(img_path)
                s_rgb.add(rh)
                rgb_hashes.add(rh)
        per_split[name] = {
            "n_examples": len(man["examples"]),
            "n_ucsf": len(s_ucsf),
            "n_qids": len(s_qids),
            "n_pages": len(s_pages),
            "n_src_hashes": len(s_src),
            "n_png_hashes": len(s_png),
            "n_rgb_hashes": len(s_rgb),
            "manifest": str(path.relative_to(root)),
            "manifest_sha256": sha256_file(path),
        }

    return {
        "ucsf": ucsf,
        "qids": qids,
        "pages": pages,
        "src_hashes": src_hashes,
        "png_hashes": png_hashes,
        "rgb_hashes": rgb_hashes,
        "per_split": per_split,
    }


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        print("ERROR: PYTHONNOUSERSITE=1 required", file=sys.stderr)
        return 2

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--target-n", type=int, default=TARGET_N)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--dataset-revision", default=DATASET_REV)
    parser.add_argument(
        "--manifest-out", default="data/manifests/docvqa_train_subset_v1.json"
    )
    parser.add_argument(
        "--answers-out", default="data/cache/docvqa_train_subset_v1_answers.json"
    )
    parser.add_argument(
        "--image-dir", default="data/images/docvqa_train_subset_v1"
    )
    parser.add_argument(
        "--audit-out", default="results/train_subset_v1_leakage_audit.json"
    )
    parser.add_argument(
        "--example-out", default="docs/examples/train_subset_v1_example.md"
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    cache_dir = str((root / ".hf_cache" / "hub").resolve())

    blocked = load_eval_blocklists(root)
    log(
        f"blocklist ucsf={len(blocked['ucsf'])} qids={len(blocked['qids'])} "
        f"pages={len(blocked['pages'])} src={len(blocked['src_hashes'])} "
        f"rgb={len(blocked['rgb_hashes'])}"
    )

    files = list_repo_files(
        args.dataset_id, repo_type="dataset", revision=args.dataset_revision
    )
    train_shards = sorted(
        f for f in files if f.startswith("data/train-") and f.endswith(".parquet")
    )
    if not train_shards:
        print("ERROR: no train shards found", file=sys.stderr)
        return 2

    exclusion_counts: Counter[str] = Counter()
    by_group: dict[tuple, list[dict]] = defaultdict(list)
    qid_to_shard: dict[int, str] = {}
    n_train_raw = 0
    n_missing_ucsf = 0
    n_no_valid_answer = 0

    # Pass 1: metadata (+ image hash only when UCSF missing for PAGE_FALLBACK keys).
    for shard in train_shards:
        path = hf_hub_download(
            args.dataset_id,
            filename=shard,
            repo_type="dataset",
            revision=args.dataset_revision,
            cache_dir=cache_dir,
        )
        meta = pq.read_table(path, columns=META_COLS).to_pydict()
        n = len(meta["questionId"])
        # Lazy image column only if this shard has missing UCSF.
        need_img_idx: list[int] = []
        for i in range(n):
            ucsf_raw = meta["ucsf_document_id"][i]
            ucsf = None if ucsf_raw is None else (str(ucsf_raw).strip() or None)
            if ucsf is None:
                need_img_idx.append(i)

        img_bytes_by_i: dict[int, bytes] = {}
        if need_img_idx:
            img_col = pq.read_table(path, columns=["image"]).to_pydict()["image"]
            for i in need_img_idx:
                img = img_col[i]
                img_bytes_by_i[i] = (
                    bytes(img["bytes"])
                    if isinstance(img, dict)
                    else bytes(img.get("bytes"))
                )

        for i in range(n):
            n_train_raw += 1
            qid = int(meta["questionId"][i])
            doc_id = int(meta["docId"][i])
            ucsf_raw = meta["ucsf_document_id"][i]
            ucsf = None if ucsf_raw is None else (str(ucsf_raw).strip() or None)
            page_raw = meta["ucsf_document_page_no"][i]
            page = None if page_raw is None else (str(page_raw).strip() or None)
            if ucsf is None:
                n_missing_ucsf += 1
            answers = [str(a) for a in list(meta["answers"][i])]
            target = first_valid_answer(answers)
            if target is None:
                n_no_valid_answer += 1
                exclusion_counts["no_valid_answer"] += 1
                continue

            if qid in blocked["qids"]:
                exclusion_counts["qid_in_eval_splits"] += 1
                continue
            if ucsf and ucsf in blocked["ucsf"]:
                exclusion_counts["ucsf_in_eval_splits"] += 1
                continue
            if page_key(ucsf, page) in blocked["pages"] and (ucsf or page):
                exclusion_counts["page_identity_in_eval_splits"] += 1
                continue

            if ucsf is None:
                src_hash = sha256_bytes(img_bytes_by_i[i])
                # Early hash exclude for PAGE_FALLBACK rows.
                if src_hash in blocked["src_hashes"]:
                    exclusion_counts["source_image_hash_in_eval_splits"] += 1
                    continue
                rgb = rgb_hash_from_bytes(img_bytes_by_i[i])
                if rgb in blocked["rgb_hashes"]:
                    exclusion_counts["rgb_hash_in_eval_splits"] += 1
                    continue
                gkey: tuple = ("PAGE_FALLBACK", doc_id, page, src_hash)
            else:
                src_hash = None  # deferred to materialization
                rgb = None
                gkey = ("UCSF", ucsf)

            row = {
                "question_id": qid,
                "doc_id": doc_id,
                "ucsf": ucsf,
                "page": page,
                "question": str(meta["question"][i]),
                "answers": answers,
                "target_answer": target,
                "source_image_sha256": src_hash,
                "rgb_hash": rgb,
                "source_shard": shard,
            }
            by_group[gkey].append(row)
            qid_to_shard[qid] = shard

        log(
            f"meta {shard} rows={n} cumulative={n_train_raw} "
            f"eligible_groups={len(by_group)} missing_ucsf_shard={len(need_img_idx)}"
        )

    group_keys = sorted(
        by_group.keys(), key=lambda k: (k[0],) + tuple(str(x) for x in k[1:])
    )
    rng = random.Random(int(args.seed))
    rng.shuffle(group_keys)

    # Oversample question IDs slightly to absorb post-hash exclusions.
    candidate_rows: list[dict] = []
    for key in group_keys:
        rows = sorted(by_group[key], key=lambda r: int(r["question_id"]))
        for row in rows:
            candidate_rows.append(row)
            if len(candidate_rows) >= int(args.target_n) + 200:
                break
        if len(candidate_rows) >= int(args.target_n) + 200:
            break

    log(f"candidates_pre_materialize={len(candidate_rows)}")

    # Pass 2: stream per-shard — hash+write immediately; never hold all images in RAM.
    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    row_by_qid = {int(r["question_id"]): r for r in candidate_rows}
    needed_by_shard: dict[str, set[int]] = defaultdict(set)
    for row in candidate_rows:
        needed_by_shard[row["source_shard"]].add(int(row["question_id"]))

    # Preserve candidate order for deterministic final cut.
    selected_meta: dict[int, dict] = {}
    for shard, qids in sorted(needed_by_shard.items()):
        if len(selected_meta) >= int(args.target_n) + 50:
            # Still process only what's needed for remaining ordered candidates.
            pass
        path = hf_hub_download(
            args.dataset_id,
            filename=shard,
            repo_type="dataset",
            revision=args.dataset_revision,
            cache_dir=cache_dir,
        )
        table = pq.read_table(path, columns=["questionId", "image"])
        qd = table.to_pydict()
        n_wrote = 0
        for i, qid_raw in enumerate(qd["questionId"]):
            qid = int(qid_raw)
            if qid not in qids:
                continue
            row = row_by_qid[qid]
            img = qd["image"][i]
            ib = (
                bytes(img["bytes"])
                if isinstance(img, dict)
                else bytes(img.get("bytes"))
            )
            src_hash = sha256_bytes(ib)
            rgb = rgb_hash_from_bytes(ib)
            if src_hash in blocked["src_hashes"]:
                exclusion_counts["source_image_hash_in_eval_splits"] += 1
                continue
            if rgb in blocked["rgb_hashes"]:
                exclusion_counts["rgb_hash_in_eval_splits"] += 1
                continue
            rel = f"{args.image_dir}/qid_{qid}.png"
            out_path = Path(rel)
            # Resume only if existing PNG pixels match pinned source (RGB hash).
            src_rgb = rgb
            if out_path.is_file():
                existing_rgb = rgb_hash_from_png(out_path)
                if existing_rgb == src_rgb:
                    png_hash = sha256_file(out_path)
                else:
                    # Mismatch: regenerate from pinned source bytes.
                    image = Image.open(io.BytesIO(ib)).convert("RGB")
                    image.save(out_path, format="PNG")
                    png_hash = sha256_file(out_path)
                    exclusion_counts["png_resume_rgb_mismatch_regenerated"] += 1
            else:
                image = Image.open(io.BytesIO(ib)).convert("RGB")
                image.save(out_path, format="PNG")
                png_hash = sha256_file(out_path)
            del ib
            if png_hash in blocked["png_hashes"]:
                exclusion_counts["png_hash_collision_after_write"] += 1
                out_path.unlink(missing_ok=True)
                continue
            selected_meta[qid] = {
                **{k: v for k, v in row.items() if not k.startswith("_")},
                "source_image_sha256": src_hash,
                "rgb_hash": rgb,
                "image_relpath": rel,
                "image_sha256_png": png_hash,
            }
            n_wrote += 1
        # Drop arrow buffers promptly.
        del table, qd
        log(f"streamed {shard}: wrote_or_kept={n_wrote} selected_so_far={len(selected_meta)}")

    out_examples: list[dict] = []
    answers_sidecar: list[dict] = []
    for row in candidate_rows:
        if len(out_examples) >= int(args.target_n):
            break
        qid = int(row["question_id"])
        meta = selected_meta.get(qid)
        if meta is None:
            continue
        gtype = "UCSF" if meta["ucsf"] else "PAGE_FALLBACK"
        out_examples.append(
            {
                "index": len(out_examples),
                "question_id": qid,
                "doc_id": int(meta["doc_id"]),
                "ucsf_document_id": meta["ucsf"] or "",
                "ucsf_document_page_no": meta["page"] or "",
                "question": meta["question"],
                "image_relpath": meta["image_relpath"],
                "image_sha256_png": meta["image_sha256_png"],
                "source_image_sha256": meta["source_image_sha256"],
                "rgb_hash": meta["rgb_hash"],
                "n_answers": len(meta["answers"]),
                "target_answer": meta["target_answer"],
                "target_selection_rule": "first_nonempty_source_listed_answer",
                "group_key_type": gtype,
                "source_shard": meta["source_shard"],
            }
        )
        answers_sidecar.append(
            {
                "question_id": qid,
                "answers": list(meta["answers"]),
                "target_answer": meta["target_answer"],
                "target_selection_rule": "first_nonempty_source_listed_answer",
            }
        )

    if len(out_examples) < int(args.target_n):
        log(f"WARNING: only selected {len(out_examples)} < target {args.target_n}")

    for i, e in enumerate(out_examples):
        e["index"] = i

    sel_ucsf = {e["ucsf_document_id"] for e in out_examples if e["ucsf_document_id"]}
    sel_qids = {e["question_id"] for e in out_examples}
    sel_pages = {
        (e["ucsf_document_id"], e["ucsf_document_page_no"]) for e in out_examples
    }
    sel_src = {e["source_image_sha256"] for e in out_examples}
    sel_rgb = {e["rgb_hash"] for e in out_examples}
    sel_png = {e["image_sha256_png"] for e in out_examples}

    overlap = {
        "ucsf": sorted(sel_ucsf & blocked["ucsf"]),
        "qids": sorted(sel_qids & blocked["qids"]),
        "pages": [list(p) for p in sorted(sel_pages & blocked["pages"])],
        "source_image_sha256": sorted(sel_src & blocked["src_hashes"]),
        "rgb_hash": sorted(sel_rgb & blocked["rgb_hashes"]),
        "png_hash": sorted(sel_png & blocked["png_hashes"]),
    }
    n_overlap = sum(len(v) for v in overlap.values())

    manifest = {
        "name": "docvqa_train_subset_v1",
        "role": "train_subset",
        "n_examples": len(out_examples),
        "target_n": int(args.target_n),
        "seed": int(args.seed),
        "dataset_id": args.dataset_id,
        "dataset_revision": args.dataset_revision,
        "split": "train",
        "sampling_procedure": (
            "1) Metadata-scan all pinned train parquet shards. "
            "2) Drop rows with no nonempty answer. "
            "3) Exclude qid / UCSF / exact page identity present in smoke, "
            "development, or holdout_v2. "
            "4) Group remaining by UCSF when present else PAGE_FALLBACK("
            "doc_id, page, source_image_sha256) — image hashed only for missing UCSF. "
            "5) Sort group keys lexicographically, shuffle groups with "
            f"random.Random({args.seed}), take groups in shuffled order; "
            "within each group take examples sorted by question_id until "
            f"target_n={args.target_n} (+buffer). "
            "6) Materialize images for candidates; exclude source/RGB hash "
            "collisions with eval splits; write PNG + sidecars. "
            "7) Selection independent of any model predictions."
        ),
        "target_selection_rule": "first_nonempty_source_listed_answer",
        "grouping_policy": (
            "UCSF source-document ID when present; else PAGE_FALLBACK("
            "doc_id, page, source_image_sha256). Missing UCSF IDs are never "
            "collapsed into one empty-string group."
        ),
        "excluded_eval_splits": ["smoke", "dev", "holdout_v2"],
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_train_shards": len(train_shards),
        "n_train_raw_rows_scanned": n_train_raw,
        "n_missing_ucsf_in_train_scanned": n_missing_ucsf,
        "n_groups_eligible": len(by_group),
        "n_ucsf_selected": len(sel_ucsf),
        "n_page_fallback_selected": sum(
            1 for e in out_examples if e["group_key_type"] == "PAGE_FALLBACK"
        ),
        "examples": out_examples,
    }
    examples_blob = json.dumps(out_examples, sort_keys=True, separators=(",", ":"))
    manifest["examples_content_sha256"] = sha256_bytes(examples_blob.encode())

    man_path = Path(args.manifest_out)
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_text = json.dumps(manifest, indent=2) + "\n"
    man_path.write_text(man_text)
    manifest_file_sha = sha256_bytes(man_text.encode())

    ans_path = Path(args.answers_out)
    ans_path.parent.mkdir(parents=True, exist_ok=True)
    ans_obj = {
        "name": "docvqa_train_subset_v1_answers",
        "n": len(answers_sidecar),
        "target_selection_rule": "first_nonempty_source_listed_answer",
        "answers": answers_sidecar,
    }
    ans_path.write_text(json.dumps(ans_obj, indent=2) + "\n")

    audit = {
        "task": "train_subset_v1_leakage_audit",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset_id": args.dataset_id,
        "dataset_revision": args.dataset_revision,
        "seed": int(args.seed),
        "target_n": int(args.target_n),
        "n_selected": len(out_examples),
        "n_train_raw_rows_scanned": n_train_raw,
        "n_train_shards": len(train_shards),
        "n_missing_ucsf_in_train_scanned": n_missing_ucsf,
        "n_no_valid_answer": n_no_valid_answer,
        "n_groups_eligible": len(by_group),
        "exclusion_counts": dict(exclusion_counts),
        "eval_blocklist_sizes": {
            "ucsf": len(blocked["ucsf"]),
            "qids": len(blocked["qids"]),
            "pages": len(blocked["pages"]),
            "src_hashes": len(blocked["src_hashes"]),
            "png_hashes": len(blocked["png_hashes"]),
            "rgb_hashes": len(blocked["rgb_hashes"]),
        },
        "eval_split_inventory": blocked["per_split"],
        "overlap_after_selection": overlap,
        "n_overlap_signals": n_overlap,
        "leakage_detected": n_overlap > 0,
        "manifest_path": str(man_path),
        "manifest_sha256": manifest_file_sha,
        "examples_content_sha256": manifest["examples_content_sha256"],
        "answers_path": str(ans_path),
        "answers_sha256": sha256_file(ans_path),
        "image_dir": str(image_dir),
        "target_selection_rule": "first_nonempty_source_listed_answer",
        "sampling_seed": int(args.seed),
        "notes": (
            "Training subset prepared for future adaptation only. "
            "No training loop executed. Development/holdout answers unused "
            "as training targets. Selection independent of model predictions."
        ),
    }
    audit_path = Path(args.audit_out)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")

    ex0 = out_examples[0]
    ans0 = next(a for a in answers_sidecar if a["question_id"] == ex0["question_id"])
    example_md = f"""# Training-subset example (not an evaluation example)

**Role:** documented real supervised training input/target from `docvqa_train_subset_v1`.  
**Not** part of smoke / development / holdout galleries.

| Field | Value |
|-------|-------|
| question_id | `{ex0["question_id"]}` |
| doc_id | `{ex0["doc_id"]}` |
| ucsf_document_id | `{ex0["ucsf_document_id"]}` |
| ucsf_document_page_no | `{ex0["ucsf_document_page_no"]}` |
| image_relpath | `{ex0["image_relpath"]}` |
| source_image_sha256 | `{ex0["source_image_sha256"][:16]}…` |
| group_key_type | `{ex0["group_key_type"]}` |
| source_shard | `{ex0["source_shard"]}` |

**Question**

{ex0["question"]}

**Supervised target** (`first_nonempty_source_listed_answer`)

`{ex0["target_answer"]}`

**All source-listed answers** (preserved in local sidecar; not used for selection beyond the target rule)

{json.dumps(ans0["answers"], ensure_ascii=False)}

**Provenance**

- Dataset: `{args.dataset_id}` revision `{args.dataset_revision}` (train split)
- Manifest: `{man_path}` sha256 `{manifest_file_sha[:16]}…`
- Seed: `{args.seed}`; target_n: `{args.target_n}`; selected: `{len(out_examples)}`
"""
    ex_path = Path(args.example_out)
    ex_path.parent.mkdir(parents=True, exist_ok=True)
    ex_path.write_text(example_md)

    log(
        json.dumps(
            {
                "n_selected": len(out_examples),
                "leakage_detected": n_overlap > 0,
                "manifest": str(man_path),
                "manifest_sha256": manifest_file_sha,
                "audit": str(audit_path),
                "example_doc": str(ex_path),
                "exclusion_counts": dict(exclusion_counts),
            },
            indent=2,
        )
    )
    return 0 if n_overlap == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
