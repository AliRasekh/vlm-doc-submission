#!/usr/bin/env python3
"""Run clean / white / unrelated visual-input conditions on InternVL3-1B base."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from . import CONDITIONS
from .paths import (
    ANSWERS,
    MANIFEST,
    MODEL_CONFIG,
    REPO_ROOT,
    VD_ART,
    assert_under_vd_artifacts,
    condition_paths,
)


def _ensure_paths() -> None:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    sys.path.insert(0, str(REPO_ROOT / "companion" / "pascal"))


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "UNKNOWN"


def _load_answers() -> dict[int, list[str]]:
    raw = json.loads(ANSWERS.read_text())
    out: dict[int, list[str]] = {}
    for item in raw.get("answers") or []:
        out[int(item["question_id"])] = list(item.get("answers") or [])
    return out


def _white_image(size_wh: tuple[int, int]) -> Image.Image:
    w, h = int(size_wh[0]), int(size_wh[1])
    return Image.new("RGB", (w, h), color=(255, 255, 255))


def run() -> int:
    _ensure_paths()
    import torch
    import yaml
    from vlm_doc.adapters.internvl import load_internvl, run_one_internvl
    from vlm_doc.metrics import SCORER_VERSION_PRIMARY, aggregate_scores, score_example, scorer_source_hash
    from vlm_doc.run_fingerprint import (
        build_scoring_provenance,
        sha256_file,
        verify_example_image,
        verify_manifest_hash,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mapping",
        default="companion/pascal/artifacts/visual_dep/unrelated_mapping.json",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA required; refuse login-node / CPU inference", file=sys.stderr)
        return 2

    mapping_path = assert_under_vd_artifacts(args.mapping, kind="mapping")
    if not mapping_path.is_file():
        print(f"ERROR: missing mapping {mapping_path}", file=sys.stderr)
        return 2
    mapping_obj = json.loads(mapping_path.read_text())
    mapping = mapping_obj["mapping"]

    cfg = yaml.safe_load(MODEL_CONFIG.read_text())
    man = json.loads(MANIFEST.read_text())
    man_v = verify_manifest_hash(man, path=MANIFEST)
    if not man_v.get("ok"):
        print(f"ERROR: manifest verify failed: {man_v}", file=sys.stderr)
        return 2
    examples = list(man.get("examples") or [])
    if args.limit is not None:
        examples = examples[: int(args.limit)]
    answers_by_qid = _load_answers()
    requested_qids = [int(ex["question_id"]) for ex in examples]

    model_id = cfg["model_id"]
    revision = cfg["revision"]
    instruction = cfg["instruction"]
    max_new_tokens = int(cfg["max_new_tokens"])
    prefer_bf16 = bool(cfg.get("prefer_bf16", True))
    use_flash_attn = bool(cfg.get("use_flash_attn", False))
    max_num = int(cfg.get("max_num", 12))
    use_thumbnail = bool(cfg.get("use_thumbnail", True))

    print("LOADING", model_id, revision, flush=True)
    loaded = load_internvl(
        model_id,
        revision,
        prefer_bf16=prefer_bf16,
        use_flash_attn=use_flash_attn,
        max_num=max_num,
        use_thumbnail=use_thumbnail,
    )
    gpu_name = torch.cuda.get_device_name(0)
    runtime = {
        "hostname": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu_name": gpu_name,
        "dtype": str(loaded.dtype).replace("torch.", ""),
        "bf16_supported": bool(loaded.bf16_supported),
        "prefer_bf16": prefer_bf16,
        "use_flash_attn": use_flash_attn,
        "attention_backend": "eager" if not use_flash_attn else "flash_attn",
        "max_num": max_num,
        "use_thumbnail": use_thumbnail,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "instruction": instruction,
        "model_id": model_id,
        "model_revision": revision,
        "source_revision": _git_sha(),
        "mapping_sha256": mapping_obj.get("mapping_sha256"),
        "config_sha256": sha256_file(MODEL_CONFIG),
        "manifest_content_sha256": man_v.get("canonical_content_sha256"),
    }
    (VD_ART / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")

    # Warm-up on first clean image
    warm_ex = examples[0]
    warm_img = Image.open(REPO_ROOT / warm_ex["image_relpath"]).convert("RGB")
    _ = run_one_internvl(
        loaded,
        image=warm_img,
        question=warm_ex["question"],
        max_new_tokens=max_new_tokens,
        instruction=instruction,
    )
    del warm_img

    for condition in CONDITIONS:
        paths = condition_paths(condition)
        assert_under_vd_artifacts(paths["jsonl"], kind="jsonl")
        paths["root"].mkdir(parents=True, exist_ok=True)
        # Clean output paths: refuse resume into a partial incompatible file.
        if paths["jsonl"].is_file() or paths["summary"].is_file():
            print(
                f"ERROR: refusing to overwrite existing {condition} outputs under {paths['root']}",
                file=sys.stderr,
            )
            return 3

        records: list[dict[str, Any]] = []
        with paths["jsonl"].open("w") as fout:
            for ex in examples:
                qid = int(ex["question_id"])
                img_v = verify_example_image(ex, root=REPO_ROOT)
                if not img_v.get("ok"):
                    print(f"ERROR: image verify failed qid={qid}: {img_v}", file=sys.stderr)
                    return 2

                orig = Image.open(REPO_ROOT / ex["image_relpath"]).convert("RGB")
                orig_wh = list(orig.size)
                donor_meta = None
                if condition == "clean":
                    image = orig
                    image_rel = ex["image_relpath"]
                    image_sha = ex.get("image_sha256_png")
                elif condition == "white":
                    image = _white_image(tuple(orig.size))
                    image_rel = f"synthetic:white:{orig_wh[0]}x{orig_wh[1]}"
                    image_sha = None
                    orig.close()
                elif condition == "unrelated":
                    m = mapping[str(qid)]
                    donor_path = REPO_ROOT / m["donor_image_relpath"]
                    image = Image.open(donor_path).convert("RGB")
                    image_rel = m["donor_image_relpath"]
                    image_sha = m.get("donor_image_sha256_png")
                    donor_meta = {
                        "donor_qid": m["donor_qid"],
                        "donor_size_wh": m["donor_size_wh"],
                        "query_size_wh": m["query_size_wh"],
                        "aspect_log_distance": m["aspect_log_distance"],
                        "resized_donor_to_query_geometry": False,
                    }
                    orig.close()
                else:
                    raise RuntimeError(condition)

                record: dict[str, Any] = {
                    "question_id": qid,
                    "condition": condition,
                    "doc_id": ex.get("doc_id"),
                    "ucsf_document_id": ex.get("ucsf_document_id"),
                    "ucsf_document_page_no": ex.get("ucsf_document_page_no"),
                    "question": ex["question"],
                    "image_relpath_used": image_rel,
                    "image_sha256_png_used": image_sha,
                    "original_image_relpath": ex["image_relpath"],
                    "original_size_wh": orig_wh,
                    "donor_meta": donor_meta,
                    "model_id": model_id,
                    "model_revision": revision,
                    "status": "ok",
                    "error": None,
                    "runtime": {
                        "gpu_name": gpu_name,
                        "dtype": runtime["dtype"],
                        "slurm_job_id": runtime["slurm_job_id"],
                        "hostname": runtime["hostname"],
                    },
                }
                t0 = time.perf_counter()
                try:
                    out = run_one_internvl(
                        loaded,
                        image=image,
                        question=ex["question"],
                        max_new_tokens=max_new_tokens,
                        instruction=instruction,
                    )
                    record.update(out)
                    info = out.get("image_info") or {}
                    record["actual_tile_count"] = int(info.get("tile_count", -1))
                    record["visual_token_count"] = info.get("visual_token_count")
                    refs = answers_by_qid.get(qid)
                    if refs is None:
                        raise RuntimeError(f"Missing references for qid={qid}")
                    # Refs for scoring only — never passed into generation.
                    record["reference_answers"] = refs
                    record["metrics"] = score_example(
                        out.get("prediction"),
                        refs,
                        status="ok",
                        primary=SCORER_VERSION_PRIMARY,
                    )
                    record["scoring_provenance"] = build_scoring_provenance(
                        primary_metric=SCORER_VERSION_PRIMARY,
                        answer_sidecar_sha256=sha256_file(ANSWERS),
                        scorer_source_sha256=scorer_source_hash(),
                    )
                    print(
                        f"{condition} ok qid={qid} tiles={record['actual_tile_count']} "
                        f"pred={out.get('prediction')!r}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    record["status"] = "error"
                    record["error"] = repr(exc)
                    refs = answers_by_qid.get(qid, [])
                    record["reference_answers"] = refs
                    record["metrics"] = score_example(
                        None, refs, status="error", primary=SCORER_VERSION_PRIMARY
                    )
                    print(f"{condition} ERROR qid={qid}: {exc}", file=sys.stderr)
                finally:
                    try:
                        image.close()
                    except Exception:
                        pass
                record["wall_seconds"] = time.perf_counter() - t0
                fout.write(json.dumps(record) + "\n")
                fout.flush()
                records.append(record)

        # Integrity: duplicates / unexpected
        qids = [int(r["question_id"]) for r in records]
        if len(qids) != len(set(qids)):
            print(f"ERROR: duplicate QIDs in {condition}", file=sys.stderr)
            return 2
        unexpected = sorted(set(qids) - set(requested_qids))
        if unexpected:
            print(f"ERROR: unexpected QIDs in {condition}: {unexpected[:10]}", file=sys.stderr)
            return 2

        agg = aggregate_scores(
            records,
            n_requested=len(requested_qids),
            required_question_ids=requested_qids,
        )
        tile_counts = [
            int(r["actual_tile_count"])
            for r in records
            if r.get("status") == "ok" and r.get("actual_tile_count") is not None
        ]
        tile_hist = Counter(tile_counts)
        n_cap = sum(1 for r in records if r.get("generation_cap_reached"))
        summary = {
            "task": "pascal_visual_dependence_condition",
            "condition": condition,
            "scores": agg,
            "n_generation_cap_hits": n_cap,
            "tile_count_histogram": {str(k): tile_hist[k] for k in sorted(tile_hist)},
            "tile_count_mean": (sum(tile_counts) / len(tile_counts)) if tile_counts else None,
            "runtime": runtime,
            "mapping_sha256": mapping_obj.get("mapping_sha256")
            if condition == "unrelated"
            else None,
            "n_records": len(records),
            "n_duplicate_qids": 0,
            "n_unexpected_qids": 0,
        }
        paths["summary"].write_text(json.dumps(summary, indent=2) + "\n")
        print(
            f"CONDITION_DONE {condition} ANLS={agg['mean_anls_requested']:.4f} "
            f"EM={agg['mean_exact_match_requested']:.4f} "
            f"ok={agg['n_completed_ok']} fail={agg['n_failed']} miss={agg['n_missing']}",
            flush=True,
        )

    done = {
        "task": "pascal_visual_dependence_done",
        "conditions": list(CONDITIONS),
        "runtime": runtime,
    }
    (VD_ART / "main_done.json").write_text(json.dumps(done, indent=2) + "\n")
    print("VISUAL_DEP_DONE", json.dumps(done), flush=True)
    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
