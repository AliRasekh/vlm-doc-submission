"""Condition fingerprints for P1 (cap-specific; no cross-cap resume)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

from vlm_doc.run_fingerprint import (
    build_inference_fingerprint,
    hash_paths,
    sha256_file,
)


def build_p1_fingerprint(
    *,
    repo_root: Path,
    cfg: dict[str, Any],
    cfg_path: Path,
    max_num: int,
    manifest_content_sha256: str,
    resolved_dtype: str,
    attention_backend: str,
    dtype_policy: str,
    source_revision: str,
) -> dict[str, Any]:
    processor_settings = {
        "adapter": "internvl3",
        "image_size": int(cfg.get("image_size", 448)),
        "max_num": int(max_num),
        "min_num": int(cfg.get("min_num", 1)),
        "use_thumbnail": bool(cfg.get("use_thumbnail", True)),
        "dynamic_image_size": True,
        "recipe": "official_dynamic_preprocess_448_thumbnail",
        "attention_backend": attention_backend,
        "experiment": "pascal_p1_visual_budget",
        "dtype_policy": dtype_policy,
        "source_git_revision": source_revision,
    }
    inference_sources = [
        "src/vlm_doc/adapters/internvl.py",
        "src/vlm_doc/adapters/internvl_conv.py",
        "src/vlm_doc/model.py",
        "companion/pascal/p1/load_wrap.py",
        "companion/pascal/p1/infer_wrap.py",
        "companion/pascal/p1/runner.py",
        "companion/pascal/p1/fingerprint.py",
        str(cfg_path.relative_to(repo_root).as_posix()),
    ]
    software = {
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "transformers": __import__("transformers").__version__,
        "python": sys.version.split()[0],
        "attention_backend": attention_backend,
        "experiment": "pascal_p1_visual_budget",
    }
    fp = build_inference_fingerprint(
        model_id=str(cfg["model_id"]),
        model_revision=str(cfg["revision"]),
        manifest_content_sha256=manifest_content_sha256,
        instruction=str(cfg["instruction"]),
        max_new_tokens=int(cfg["max_new_tokens"]),
        do_sample=bool(cfg.get("do_sample", False)),
        prefer_bf16=bool(cfg.get("prefer_bf16", True)),
        seed=int(cfg.get("seed", 0)),
        config_path=str(cfg_path.relative_to(repo_root).as_posix()),
        config_sha256=sha256_file(cfg_path),
        resolved_dtype=resolved_dtype,
        processor_settings=processor_settings,
        software=software,
        inference_source_hashes=hash_paths([repo_root / p for p in inference_sources]),
        adapter_dir=None,
    )
    # Explicit cap echo for logs / resume diagnostics
    fp["p1_max_num"] = int(max_num)
    fp["p1_source_git_revision"] = source_revision
    return fp


def fingerprint_sha(fp: dict[str, Any]) -> str:
    return str(fp["run_fingerprint_sha256"])


def dumps_compact(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)
