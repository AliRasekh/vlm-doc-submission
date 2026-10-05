#!/usr/bin/env python3
"""Audit source-document / page / hash overlap across frozen splits.

Fails explicitly (exit 1) when UCSF source-document, page-identity, source-file
hash, or decoded RGB pixel hash overlap is detected between holdout and
development/smoke.

Grouping:
  - Prefer ``ucsf_document_id`` when present and nonempty.
  - Missing UCSF uses page-level fallback
    ``(PAGE_FALLBACK, doc_id, page, source_image_sha256)`` — never collapses
    unknowns into one empty-string group.
  - ``doc_id`` alone is NOT a reliable whole-PDF grouping key.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def rgb_hash_from_png(path: Path) -> str:
    im = Image.open(path).convert("RGB")
    return sha256_bytes(im.tobytes() + f"{im.size[0]}x{im.size[1]}".encode())


def load_split(manifest_path: Path, root: Path) -> dict:
    man = json.loads(manifest_path.read_text())
    ucsfs: set[str] = set()
    page_ids: set[tuple] = set()
    src_hashes: set[str] = set()
    rgb_hashes: set[str] = set()
    docs: set[int] = set()
    missing_ucsf = 0
    for e in man["examples"]:
        docs.add(int(e["doc_id"]))
        ucsf = e.get("ucsf_document_id")
        ucsf_s = None if ucsf is None else str(ucsf).strip() or None
        page = e.get("ucsf_document_page_no")
        page_s = None if page is None else str(page).strip() or None
        src = e.get("source_image_sha256") or e.get("image_sha256_png")
        img_path = root / e["image_relpath"]
        rgb = rgb_hash_from_png(img_path) if img_path.is_file() else None
        if src:
            src_hashes.add(src)
        if rgb:
            rgb_hashes.add(rgb)
        if ucsf_s is None:
            missing_ucsf += 1
            # page-level fallback — never empty-string collapse
            key = ("PAGE_FALLBACK", int(e["doc_id"]), page_s, src)
            page_ids.add(key)
        else:
            ucsfs.add(ucsf_s)
            page_ids.add((ucsf_s, page_s))
    return {
        "name": man.get("name"),
        "path": str(manifest_path),
        "n": len(man["examples"]),
        "ucsfs": ucsfs,
        "docs": docs,
        "page_ids": page_ids,
        "src_hashes": src_hashes,
        "rgb_hashes": rgb_hashes,
        "missing_ucsf": missing_ucsf,
    }


def main() -> int:
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        print("ERROR: PYTHONNOUSERSITE=1 required in launcher", file=sys.stderr)
        return 2

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", default="data/manifests/docvqa_smoke_v1.json")
    parser.add_argument("--dev", default="data/manifests/docvqa_dev_v1.json")
    parser.add_argument("--holdout", required=True)
    parser.add_argument("--out", default="results/split_leakage_check.json")
    parser.add_argument(
        "--fail-on-overlap",
        action="store_true",
        default=True,
        help="Exit 1 when any holdout↔(dev|smoke) overlap is found (default).",
    )
    parser.add_argument("--no-fail-on-overlap", action="store_true")
    args = parser.parse_args()
    fail = not args.no_fail_on_overlap

    root = Path(__file__).resolve().parents[1]
    os.chdir(root)

    S = load_split(Path(args.smoke), root)
    D = load_split(Path(args.dev), root)
    H = load_split(Path(args.holdout), root)

    report = {
        "holdout": H["name"],
        "dev": D["name"],
        "smoke": S["name"],
        "counts": {
            "dev_ucsf": len(D["ucsfs"]),
            "hold_ucsf": len(H["ucsfs"]),
            "smoke_ucsf": len(S["ucsfs"]),
            "dev_missing_ucsf": D["missing_ucsf"],
            "hold_missing_ucsf": H["missing_ucsf"],
            "smoke_missing_ucsf": S["missing_ucsf"],
        },
        "overlap_ucsf_dev_hold": sorted(D["ucsfs"] & H["ucsfs"]),
        "overlap_ucsf_smoke_hold": sorted(S["ucsfs"] & H["ucsfs"]),
        "overlap_ucsf_smoke_dev": sorted(S["ucsfs"] & D["ucsfs"]),
        "overlap_page_ids_dev_hold": [list(x) for x in sorted(D["page_ids"] & H["page_ids"])],
        "overlap_page_ids_smoke_hold": [
            list(x) for x in sorted(S["page_ids"] & H["page_ids"])
        ],
        "overlap_src_hash_dev_hold": sorted(D["src_hashes"] & H["src_hashes"]),
        "overlap_rgb_hash_dev_hold": sorted(D["rgb_hashes"] & H["rgb_hashes"]),
        "overlap_src_hash_smoke_hold": sorted(S["src_hashes"] & H["src_hashes"]),
        "overlap_rgb_hash_smoke_hold": sorted(S["rgb_hashes"] & H["rgb_hashes"]),
        "grouping_policy": (
            "UCSF source-document ID when present; else PAGE_FALLBACK("
            "doc_id, page, source_image_sha256). Missing IDs must not collapse "
            "into one empty-string group. Future training samples must be checked "
            "against development and holdout UCSF IDs and page hashes."
        ),
    }
    n_bad = (
        len(report["overlap_ucsf_dev_hold"])
        + len(report["overlap_ucsf_smoke_hold"])
        + len(report["overlap_page_ids_dev_hold"])
        + len(report["overlap_page_ids_smoke_hold"])
        + len(report["overlap_src_hash_dev_hold"])
        + len(report["overlap_rgb_hash_dev_hold"])
        + len(report["overlap_src_hash_smoke_hold"])
        + len(report["overlap_rgb_hash_smoke_hold"])
    )
    report["n_overlap_signals"] = n_bad
    report["leakage_detected"] = n_bad > 0

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("counts", "n_overlap_signals", "leakage_detected", "overlap_ucsf_dev_hold")}, indent=2))
    if n_bad > 0:
        msg = f"leakage detected ({n_bad} overlap signals). See {args.out}"
        if fail:
            print(f"FAIL: {msg}", file=sys.stderr)
            return 1
        print(f"WARNING (non-fatal): {msg}", file=sys.stderr)
        return 0
    print(f"OK: no holdout overlap with development/smoke. wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
