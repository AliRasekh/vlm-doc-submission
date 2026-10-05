#!/usr/bin/env python3
"""Download SmolVLM metadata + one required weight file into a persistent HF cache."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default="HuggingFaceTB/SmolVLM-256M-Instruct",
    )
    parser.add_argument(
        "--revision",
        default="7e3e67edbbed1bf9888184d9df282b700a323964",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("HF_HOME", ".hf_cache"),
        help="Persistent Hugging Face cache root (HF_HOME).",
    )
    parser.add_argument(
        "--out-json",
        default="results/model_download_smolvlm256m.json",
    )
    args = parser.parse_args()

    cache_root = Path(args.cache_dir).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root)
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HUB_CACHE"])

    from huggingface_hub import hf_hub_download, model_info

    info = model_info(args.model_id, revision=args.revision)
    sha = info.sha
    print(f"model_id={args.model_id}")
    print(f"requested_revision={args.revision}")
    print(f"resolved_sha={sha}")

    # Metadata files first
    meta_files = [
        "config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
    ]
    downloaded = {}
    for name in meta_files:
        try:
            path = hf_hub_download(
                args.model_id,
                filename=name,
                revision=args.revision,
                cache_dir=os.environ["HF_HUB_CACHE"],
            )
            downloaded[name] = {"path": path, "status": "ok"}
            print(f"downloaded_meta {name}")
        except Exception as exc:  # noqa: BLE001
            downloaded[name] = {"status": "error", "error": repr(exc)}
            print(f"meta_skip {name}: {exc}")

    # One required weight file
    weight_name = "model.safetensors"
    try:
        weight_path = hf_hub_download(
            args.model_id,
            filename=weight_name,
            revision=args.revision,
            cache_dir=os.environ["HF_HUB_CACHE"],
        )
        weight_size = Path(weight_path).stat().st_size
        downloaded[weight_name] = {
            "path": weight_path,
            "status": "ok",
            "bytes": weight_size,
        }
        print(f"downloaded_weight {weight_name} bytes={weight_size}")
    except Exception as exc:  # noqa: BLE001
        downloaded[weight_name] = {"status": "error", "error": repr(exc)}
        print(f"WEIGHT_DOWNLOAD_FAILED: {exc}", file=sys.stderr)
        out = {
            "model_id": args.model_id,
            "revision": args.revision,
            "resolved_sha": sha,
            "downloaded": downloaded,
            "ok": False,
        }
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(out, indent=2))
        return 1

    out = {
        "model_id": args.model_id,
        "revision": args.revision,
        "resolved_sha": sha,
        "downloaded": {k: {kk: vv for kk, vv in v.items() if kk != "path"} for k, v in downloaded.items()},
        "hf_home": str(cache_root),
        "ok": True,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
