#!/usr/bin/env python3
"""Build an 8-example DocVQA smoke manifest from one validation parquet shard.

Provenance:
  - Official DocVQA validation split mirrored on Hugging Face as
    HuggingFaceM4/DocumentVQA (features: questionId, question, image, docId,
    ucsf_document_id, ucsf_document_page_no, answers).
  - VLMEvalKit DocVQA_VAL (commit recorded in docs/sources.md) points at an
    OpenCompass TSV whose HTTPS endpoint currently fails TLS verification
    (expired certificate). That route is documented but not used here.
  - This script downloads a single validation parquet shard only (not train).

Selection (model-independent):
  1. Load one shard (default: validation-00000-of-00017.parquet).
  2. Sort rows by integer questionId ascending.
  3. Take 8 evenly spaced indices: round(i * (n-1) / 7) for i in 0..7.
  4. Persist images under data/images/ (gitignored) and a tracked JSON manifest
     of identifiers + content hashes (no answer text in the tracked file's
     answer field is optional — answers stored in a local gitignored sidecar).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image
import io


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def select_indices(n: int, k: int = 8) -> list[int]:
    if n < k:
        raise ValueError(f"Need at least {k} rows, found {n}")
    if k == 1:
        return [0]
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default="HuggingFaceM4/DocumentVQA")
    parser.add_argument(
        "--dataset-revision",
        default="a44195f8f989c10d06ab4ccc98ffe9cdb6d928e4",
    )
    parser.add_argument(
        "--shard",
        default="data/validation-00000-of-00017.parquet",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("HF_HOME", ".hf_cache"),
    )
    parser.add_argument("--image-dir", default="data/images/docvqa_smoke")
    parser.add_argument(
        "--manifest-out",
        default="data/manifests/docvqa_smoke_v1.json",
    )
    parser.add_argument(
        "--answers-out",
        default="data/cache/docvqa_smoke_v1_answers.json",
        help="Local gitignored sidecar with reference answers.",
    )
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()

    cache_root = Path(args.cache_dir).resolve()
    hub_cache = cache_root / "hub"
    hub_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)

    print(f"downloading shard {args.shard} from {args.dataset_id}@{args.dataset_revision}")
    shard_path = hf_hub_download(
        repo_id=args.dataset_id,
        repo_type="dataset",
        filename=args.shard,
        revision=args.dataset_revision,
        cache_dir=str(hub_cache),
    )
    shard_bytes = Path(shard_path).stat().st_size
    print(f"shard_path_ok bytes={shard_bytes}")

    table = pq.read_table(shard_path)
    df = table.to_pandas()
    df = df.sort_values("questionId", ascending=True).reset_index(drop=True)
    idxs = select_indices(len(df), args.k)
    print(f"rows_in_shard={len(df)} selected_indices={idxs}")

    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    examples = []
    answers = []
    for rank, idx in enumerate(idxs):
        row = df.iloc[idx]
        qid = int(row["questionId"])
        doc_id = int(row["docId"])
        ucsf_doc = str(row.get("ucsf_document_id", ""))
        page_no = str(row.get("ucsf_document_page_no", ""))
        question = str(row["question"])
        ans_list = list(row["answers"]) if row["answers"] is not None else []

        # Extract image bytes from parquet struct/dict
        img_obj = row["image"]
        if isinstance(img_obj, dict) and "bytes" in img_obj and img_obj["bytes"] is not None:
            img_bytes = img_obj["bytes"]
        elif hasattr(img_obj, "get"):
            img_bytes = img_obj.get("bytes")
        else:
            # PIL Image already
            buf = io.BytesIO()
            Image.fromarray(img_obj).save(buf, format="PNG")
            img_bytes = buf.getvalue()

        if not isinstance(img_bytes, (bytes, bytearray)):
            raise TypeError(f"Unexpected image payload type for qid={qid}: {type(img_bytes)}")

        img_hash = sha256_bytes(bytes(img_bytes))
        rel_name = f"qid_{qid}.png"
        out_path = image_dir / rel_name
        # Always rewrite as PNG via PIL for stable local files
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image.save(out_path, format="PNG")
        png_hash = sha256_bytes(out_path.read_bytes())

        examples.append(
            {
                "smoke_index": rank,
                "question_id": qid,
                "doc_id": doc_id,
                "ucsf_document_id": ucsf_doc,
                "ucsf_document_page_no": page_no,
                "question": question,
                "image_relpath": str(Path(args.image_dir) / rel_name),
                "image_sha256_png": png_hash,
                "source_image_sha256": img_hash,
                "shard_row_index_sorted": int(idx),
            }
        )
        answers.append(
            {
                "question_id": qid,
                "answers": [str(a) for a in ans_list],
            }
        )
        print(f"selected qid={qid} doc={doc_id} page={page_no} q={question[:60]!r}")

    manifest = {
        "name": "docvqa_smoke_v1",
        "n_examples": len(examples),
        "dataset_id": args.dataset_id,
        "dataset_revision": args.dataset_revision,
        "split": "validation",
        "shard": args.shard,
        "shard_bytes": shard_bytes,
        "selection": {
            "procedure": (
                "Sort shard by questionId ascending; take 8 evenly spaced "
                "indices round(i*(n-1)/7) for i in 0..7. Independent of model outputs."
            ),
            "k": args.k,
            "sorted_shard_indices": idxs,
        },
        "holdout_note": (
            "These question_id / document pages must be excluded from final holdout evaluation."
        ),
        "vlmevalkit_note": (
            "VLMEvalKit DocVQA_VAL TSV at opencompass.openxlab.space failed HTTPS "
            "(expired certificate) during Phase 1; HuggingFaceM4/DocumentVQA validation "
            "is used as an accessible official DocVQA validation mirror."
        ),
        "examples": examples,
    }

    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    answers_path = Path(args.answers_out)
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    answers_path.write_text(json.dumps({"answers": answers}, indent=2) + "\n")

    print(f"wrote {manifest_path}")
    print(f"wrote answers sidecar {answers_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
