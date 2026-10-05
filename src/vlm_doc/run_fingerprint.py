"""Manifest hashing, image verification, and inference-run fingerprints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_manifest_content_hash(manifest: dict[str, Any]) -> str:
    """Hash manifest content excluding self-hash field(s)."""
    body = {
        k: v
        for k, v in manifest.items()
        if k not in ("manifest_sha256", "content_sha256")
    }
    canonical = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return sha256_text(canonical + "\n")


def verify_manifest_hash(
    manifest: dict[str, Any], *, path: Path | None = None
) -> dict[str, Any]:
    content_hash = canonical_manifest_content_hash(manifest)
    stored = manifest.get("manifest_sha256") or manifest.get("content_sha256")
    raw_file_sha256 = (
        sha256_file(path) if path is not None and path.is_file() else None
    )
    ok = stored == content_hash
    return {
        "ok": ok,
        "canonical_content_sha256": content_hash,
        "stored_manifest_sha256": stored,
        "raw_file_sha256": raw_file_sha256,
        "note": (
            "canonical_content_sha256 excludes the self-hash field and uses "
            "sort_keys JSON; raw_file_sha256 is the on-disk file digest and may differ."
        ),
    }


def verify_example_image(example: dict[str, Any], *, root: Path) -> dict[str, Any]:
    rel = example.get("image_relpath")
    if not rel:
        return {"ok": False, "error": "missing image_relpath"}
    path = root / rel
    if not path.is_file():
        return {"ok": False, "error": f"missing image file: {rel}"}
    digest = sha256_file(path)
    expected = example.get("image_sha256_png") or example.get("source_image_sha256")
    ok = expected is None or digest == expected
    return {
        "ok": ok,
        "path": rel,
        "sha256": digest,
        "expected_sha256": expected,
        "mismatch": (not ok),
    }


def hash_paths(paths: Sequence[str | Path]) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in paths:
        path = Path(p)
        out[str(path.as_posix())] = (
            sha256_file(path) if path.is_file() else "MISSING"
        )
    return out


def build_inference_fingerprint(
    *,
    model_id: str,
    model_revision: str,
    manifest_content_sha256: str,
    instruction: str,
    max_new_tokens: int,
    do_sample: bool,
    prefer_bf16: bool,
    seed: int,
    config_path: str,
    config_sha256: str,
    resolved_dtype: str,
    processor_settings: dict[str, Any],
    software: dict[str, str],
    inference_source_hashes: dict[str, str],
    adapter_dir: str | None = None,
    adapter_config_sha256: str | None = None,
    adapter_weights_sha256: dict[str, str] | None = None,
    train_update: int | None = None,
    perturbation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fingerprint for inference reuse (excludes scoring code / docs)."""
    payload = {
        "model_id": model_id,
        "model_revision": model_revision,
        "manifest_content_sha256": manifest_content_sha256,
        "instruction": instruction,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "prefer_bf16": bool(prefer_bf16),
        "seed": int(seed),
        "config_path": config_path,
        "config_sha256": config_sha256,
        "resolved_dtype": resolved_dtype,
        "processor_settings": processor_settings,
        "software": software,
        "inference_source_hashes": inference_source_hashes,
        "adapter_dir": adapter_dir,
        "adapter_config_sha256": adapter_config_sha256,
        "adapter_weights_sha256": adapter_weights_sha256,
        "train_update": train_update,
        # When set, prevents clean predictions from being resumed into corrupted runs.
        "perturbation": perturbation,
        "fingerprint_schema": "inference_v2",
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return {
        **payload,
        "run_fingerprint_sha256": sha256_text(canonical),
    }


def build_scoring_provenance(
    *,
    primary_metric: str,
    answer_sidecar_sha256: str,
    scorer_source_sha256: str,
) -> dict[str, Any]:
    return {
        "primary_metric": primary_metric,
        "answer_sidecar_sha256": answer_sidecar_sha256,
        "scorer_source_sha256": scorer_source_sha256,
        "scoring_schema": "scoring_v2",
    }


def load_compatible_done_ids(
    jsonl_path: Path,
    expected_fingerprint: str,
) -> tuple[set[int], int, int]:
    done: set[int] = set()
    n_ok = 0
    n_incompat = 0
    if not jsonl_path.is_file():
        return done, 0, 0
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            fp = (rec.get("run_fingerprint") or {}).get("run_fingerprint_sha256")
            if fp != expected_fingerprint:
                n_incompat += 1
                continue
            if rec.get("status") == "ok" and "question_id" in rec:
                qid = int(rec["question_id"])
                if qid not in done:
                    done.add(qid)
                    n_ok += 1
    return done, n_ok, n_incompat


def assert_output_fingerprint_coherent(
    jsonl_path: Path, expected_fingerprint: str
) -> None:
    """Reject mixed-fingerprint or wrong-fingerprint output files."""
    if not jsonl_path.is_file():
        return
    seen: set[str] = set()
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            fp = (rec.get("run_fingerprint") or {}).get("run_fingerprint_sha256")
            if fp:
                seen.add(fp)
    if not seen:
        return
    if len(seen) > 1 or (expected_fingerprint not in seen):
        raise RuntimeError(
            f"Incompatible output file {jsonl_path}: fingerprints={sorted(seen)} "
            f"expected={expected_fingerprint}. Use a new --out-jsonl path."
        )
