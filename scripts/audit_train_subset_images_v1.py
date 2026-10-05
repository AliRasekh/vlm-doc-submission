#!/usr/bin/env python3
"""Audit frozen docvqa_train_subset_v1 PNG provenance + recheck leakage.

Does not resample membership. Regenerates PNGs only when RGB content mismatches
the source-derived rgb_hash (or file/png hash mismatch vs manifest). Requires
complete smoke/dev/holdout_v2 image coverage for hash-based leakage.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

DATASET_ID = "HuggingFaceM4/DocumentVQA"
DATASET_REV = "a44195f8f989c10d06ab4ccc98ffe9cdb6d928e4"


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


def examples_content_sha256(examples: list[dict]) -> str:
    blob = json.dumps(examples, sort_keys=True, separators=(",", ":"))
    return sha256_bytes(blob.encode())


def load_eval_hashes(root: Path, *, require_images: bool) -> dict:
    splits = {
        "smoke": root / "data/manifests/docvqa_smoke_v1.json",
        "dev": root / "data/manifests/docvqa_dev_v1.json",
        "holdout_v2": root / "data/manifests/docvqa_holdout_v2.json",
    }
    ucsf: set[str] = set()
    qids: set[int] = set()
    pages: set[tuple[str, str]] = set()
    src: set[str] = set()
    png: set[str] = set()
    rgb: set[str] = set()
    missing_images: list[str] = []
    per: dict = {}
    for name, path in splits.items():
        man = json.loads(path.read_text())
        n_ok = 0
        for e in man["examples"]:
            qids.add(int(e["question_id"]))
            u = str(e.get("ucsf_document_id") or "").strip()
            p = str(e.get("ucsf_document_page_no") or "").strip()
            if u:
                ucsf.add(u)
            pages.add((u, p))
            if e.get("source_image_sha256"):
                src.add(e["source_image_sha256"])
            if e.get("image_sha256_png"):
                png.add(e["image_sha256_png"])
            img = root / e["image_relpath"]
            if not img.is_file():
                missing_images.append(e["image_relpath"])
                continue
            rgb.add(rgb_hash_from_png(img))
            n_ok += 1
        per[name] = {
            "n_examples": len(man["examples"]),
            "n_images_verified": n_ok,
            "manifest_raw_file_sha256": sha256_file(path),
        }
    if require_images and missing_images:
        raise SystemExit(
            f"ERROR: missing {len(missing_images)} eval images; "
            "hash leakage audit cannot proceed. first="
            + missing_images[0]
        )
    return {
        "ucsf": ucsf,
        "qids": qids,
        "pages": pages,
        "src": src,
        "png": png,
        "rgb": rgb,
        "per_split": per,
        "missing_images": missing_images,
    }


def fetch_source_bytes(
    *,
    qid: int,
    shard: str,
    cache_dir: str,
) -> bytes:
    path = hf_hub_download(
        DATASET_ID,
        filename=shard,
        repo_type="dataset",
        revision=DATASET_REV,
        cache_dir=cache_dir,
    )
    table = pq.read_table(path, columns=["questionId", "image"])
    d = table.to_pydict()
    for i, q in enumerate(d["questionId"]):
        if int(q) == qid:
            img = d["image"][i]
            return bytes(img["bytes"]) if isinstance(img, dict) else bytes(img.get("bytes"))
    raise KeyError(f"qid {qid} not found in {shard}")


def main() -> int:
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        print("ERROR: PYTHONNOUSERSITE=1 required", file=sys.stderr)
        return 2

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default="data/manifests/docvqa_train_subset_v1.json"
    )
    parser.add_argument(
        "--out", default="results/train_subset_v1_image_integrity_audit.json"
    )
    parser.add_argument("--repair", action="store_true", default=True)
    parser.add_argument("--no-repair", action="store_false", dest="repair")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    cache_dir = str((root / ".hf_cache" / "hub").resolve())

    man_path = Path(args.manifest)
    raw_file_sha = sha256_file(man_path)
    man = json.loads(man_path.read_text())
    examples = man["examples"]
    stored_examples_sha = man.get("examples_content_sha256")
    recomputed_examples_sha = examples_content_sha256(examples)

    print(
        f"manifest_raw_file_sha256={raw_file_sha}\n"
        f"examples_content_sha256_stored={stored_examples_sha}\n"
        f"examples_content_sha256_recomputed={recomputed_examples_sha}",
        flush=True,
    )
    if stored_examples_sha != recomputed_examples_sha:
        print("ERROR: examples_content_sha256 mismatch vs examples[]", file=sys.stderr)
        return 2

    blocked = load_eval_hashes(root, require_images=True)
    print(
        f"eval_blocklist ucsf={len(blocked['ucsf'])} qids={len(blocked['qids'])} "
        f"pages={len(blocked['pages'])} src={len(blocked['src'])} "
        f"rgb={len(blocked['rgb'])}",
        flush=True,
    )

    # Deduplicate physical PNG checks by image_relpath while validating every mapping.
    by_path: dict[str, list[dict]] = defaultdict(list)
    for e in examples:
        by_path[e["image_relpath"]].append(e)

    repairs: list[dict] = []
    failures: list[dict] = []
    n_ok = 0
    unique_rgb_checked = 0

    # Group needed source fetches by shard for efficiency when repairing.
    need_source: dict[str, set[int]] = defaultdict(set)

    path_status: dict[str, dict] = {}
    for rel, mapped in by_path.items():
        path = root / rel
        ex0 = mapped[0]
        expected_png = ex0["image_sha256_png"]
        expected_rgb = ex0["rgb_hash"]
        expected_src = ex0["source_image_sha256"]
        # Consistency across mappings to same path
        for e in mapped[1:]:
            if (
                e["image_sha256_png"] != expected_png
                or e["rgb_hash"] != expected_rgb
                or e["source_image_sha256"] != expected_src
            ):
                failures.append(
                    {
                        "question_id": e["question_id"],
                        "error": "inconsistent_manifest_hashes_for_same_path",
                        "path": rel,
                    }
                )

        status = {
            "path": rel,
            "n_manifest_mappings": len(mapped),
            "exists": path.is_file(),
            "repaired": False,
        }
        if not path.is_file():
            status["action"] = "missing_needs_repair"
            need_source[ex0["source_shard"]].add(int(ex0["question_id"]))
            path_status[rel] = status
            continue

        png_sha = sha256_file(path)
        try:
            rgb = rgb_hash_from_png(path)
        except Exception as exc:  # noqa: BLE001
            status["action"] = "corrupt_needs_repair"
            status["error"] = str(exc)
            need_source[ex0["source_shard"]].add(int(ex0["question_id"]))
            path_status[rel] = status
            continue

        unique_rgb_checked += 1
        png_ok = png_sha == expected_png
        rgb_ok = rgb == expected_rgb
        status.update(
            {
                "png_sha256": png_sha,
                "rgb_hash": rgb,
                "png_match": png_ok,
                "rgb_match": rgb_ok,
            }
        )
        if png_ok and rgb_ok:
            status["action"] = "ok"
            n_ok += len(mapped)
            path_status[rel] = status
            continue
        status["action"] = "mismatch_needs_repair"
        need_source[ex0["source_shard"]].add(int(ex0["question_id"]))
        path_status[rel] = status

    if need_source and not args.repair:
        print("ERROR: repairs required but --no-repair set", file=sys.stderr)
        return 2

    # Repair from pinned train shards
    qid_to_ex = {int(e["question_id"]): e for e in examples}
    for shard, qids in sorted(need_source.items()):
        print(f"repair_fetch shard={shard} n={len(qids)}", flush=True)
        path = hf_hub_download(
            DATASET_ID,
            filename=shard,
            repo_type="dataset",
            revision=DATASET_REV,
            cache_dir=cache_dir,
        )
        table = pq.read_table(path, columns=["questionId", "image"])
        d = table.to_pydict()
        wanted = set(qids)
        for i, qraw in enumerate(d["questionId"]):
            qid = int(qraw)
            if qid not in wanted:
                continue
            ex = qid_to_ex[qid]
            img = d["image"][i]
            ib = bytes(img["bytes"]) if isinstance(img, dict) else bytes(img.get("bytes"))
            src_hash = sha256_bytes(ib)
            rgb = rgb_hash_from_bytes(ib)
            if src_hash != ex["source_image_sha256"]:
                failures.append(
                    {
                        "question_id": qid,
                        "error": "source_bytes_sha_mismatch_vs_manifest",
                        "got": src_hash,
                        "expected": ex["source_image_sha256"],
                    }
                )
                continue
            if rgb != ex["rgb_hash"]:
                failures.append(
                    {
                        "question_id": qid,
                        "error": "source_rgb_mismatch_vs_manifest",
                        "got": rgb,
                        "expected": ex["rgb_hash"],
                    }
                )
                continue
            out_path = root / ex["image_relpath"]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.open(io.BytesIO(ib)).convert("RGB").save(out_path, format="PNG")
            png_sha = sha256_file(out_path)
            rgb_after = rgb_hash_from_png(out_path)
            if rgb_after != ex["rgb_hash"]:
                failures.append(
                    {
                        "question_id": qid,
                        "error": "repaired_png_rgb_still_mismatch",
                    }
                )
                continue
            # Update manifest png hash if it changed after regeneration
            old_png = ex["image_sha256_png"]
            if png_sha != old_png:
                for e in by_path[ex["image_relpath"]]:
                    e["image_sha256_png"] = png_sha
            repairs.append(
                {
                    "question_id": qid,
                    "path": ex["image_relpath"],
                    "reason": path_status[ex["image_relpath"]].get("action"),
                    "old_png_sha256": old_png,
                    "new_png_sha256": png_sha,
                    "rgb_hash": rgb_after,
                }
            )
            path_status[ex["image_relpath"]]["repaired"] = True
            path_status[ex["image_relpath"]]["action"] = "repaired_ok"
            path_status[ex["image_relpath"]]["png_sha256"] = png_sha
            path_status[ex["image_relpath"]]["rgb_hash"] = rgb_after
            n_ok += len(by_path[ex["image_relpath"]])
            wanted.discard(qid)
        if wanted:
            for qid in sorted(wanted):
                failures.append(
                    {"question_id": qid, "error": "qid_not_found_in_source_shard"}
                )

    # Final per-example validation
    final_fail = 0
    for e in examples:
        rel = e["image_relpath"]
        path = root / rel
        if not path.is_file():
            final_fail += 1
            continue
        if sha256_file(path) != e["image_sha256_png"]:
            final_fail += 1
            failures.append(
                {
                    "question_id": e["question_id"],
                    "error": "final_png_hash_mismatch",
                }
            )
            continue
        if rgb_hash_from_png(path) != e["rgb_hash"]:
            final_fail += 1
            failures.append(
                {
                    "question_id": e["question_id"],
                    "error": "final_rgb_hash_mismatch",
                }
            )

    # Leakage recheck on selected set
    sel_ucsf = {e["ucsf_document_id"] for e in examples if e["ucsf_document_id"]}
    sel_qids = {int(e["question_id"]) for e in examples}
    sel_pages = {
        (e["ucsf_document_id"], e["ucsf_document_page_no"]) for e in examples
    }
    sel_src = {e["source_image_sha256"] for e in examples}
    sel_rgb = {e["rgb_hash"] for e in examples}
    sel_png = {e["image_sha256_png"] for e in examples}
    overlap = {
        "ucsf": sorted(sel_ucsf & blocked["ucsf"]),
        "qids": sorted(sel_qids & blocked["qids"]),
        "pages": [list(p) for p in sorted(sel_pages & blocked["pages"])],
        "source_image_sha256": sorted(sel_src & blocked["src"]),
        "rgb_hash": sorted(sel_rgb & blocked["rgb"]),
        "png_hash": sorted(sel_png & blocked["png"]),
    }
    n_overlap = sum(len(v) for v in overlap.values())

    # If PNG hashes in examples changed, rewrite manifest with updated
    # examples_content_sha256 (membership/identity fields otherwise unchanged).
    new_examples_sha = examples_content_sha256(examples)
    manifest_rewritten = False
    if new_examples_sha != stored_examples_sha:
        man["examples"] = examples
        man["examples_content_sha256"] = new_examples_sha
        man["image_integrity_audit_utc"] = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        man_path.write_text(json.dumps(man, indent=2) + "\n")
        manifest_rewritten = True
        raw_file_sha = sha256_file(man_path)

    report = {
        "task": "train_subset_v1_image_integrity_audit",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest_path": str(man_path),
        "hash_terminology": {
            "manifest_raw_file_sha256": "SHA256 of on-disk manifest file bytes",
            "examples_content_sha256": (
                "SHA256 of json.dumps(examples, sort_keys=True, separators=(',',':'))"
            ),
            "note": (
                "examples_content_sha256 is NOT the same as a full-manifest canonical "
                "hash that would exclude self-hash fields of the whole object."
            ),
        },
        "manifest_raw_file_sha256": raw_file_sha,
        "examples_content_sha256": new_examples_sha,
        "examples_content_sha256_before_audit": stored_examples_sha,
        "manifest_rewritten": manifest_rewritten,
        "n_examples": len(examples),
        "n_unique_image_paths": len(by_path),
        "n_unique_rgb_checked_before_repair": unique_rgb_checked,
        "n_repairs": len(repairs),
        "repairs": repairs,
        "n_failures": len(failures) + final_fail,
        "failures": failures[:50],
        "eval_split_inventory": blocked["per_split"],
        "leakage_overlap": overlap,
        "n_overlap_signals": n_overlap,
        "leakage_detected": n_overlap > 0,
        "ok": (len(failures) + final_fail) == 0 and n_overlap == 0,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in [
        "ok", "n_examples", "n_repairs", "n_failures", "leakage_detected",
        "manifest_rewritten", "examples_content_sha256", "manifest_raw_file_sha256",
    ]}, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
