#!/usr/bin/env python3
"""Pascal P1 visual-budget runner (development only; no holdout)."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

# Repo layout: companion/pascal/p1/runner.py -> parents[3] = repo root
REPO_ROOT = Path(__file__).resolve().parents[3]


def _ensure_paths() -> None:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    sys.path.insert(0, str(REPO_ROOT / "companion" / "pascal"))


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip()
        )
    except Exception:
        return "UNKNOWN"


def _finite_check(model: Any) -> dict[str, Any]:
    bad = []
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p is None or not torch.is_floating_point(p):
                continue
            t = p.detach()
            if not torch.isfinite(t).all():
                bad.append(name)
                if len(bad) >= 5:
                    break
    return {"ok": len(bad) == 0, "nonfinite_param_examples": bad}


def main() -> int:
    _ensure_paths()
    from p1.caps import candidate_max_nums, execution_order, omitted_caps
    from p1.fingerprint import build_p1_fingerprint, fingerprint_sha
    from p1.infer_wrap import run_one_request
    from p1.load_wrap import load_internvl_p1, set_max_num
    from vlm_doc.metrics import aggregate_scores, score_example
    from vlm_doc.run_fingerprint import (
        assert_output_fingerprint_coherent,
        load_compatible_done_ids,
        sha256_file,
        verify_example_image,
        verify_manifest_hash,
    )

    parser = argparse.ArgumentParser(description="Pascal P1 visual-budget experiment")
    parser.add_argument(
        "--config",
        default="companion/pascal/configs/p1_visual_budget.yaml",
    )
    parser.add_argument(
        "--out-dir",
        default="companion/pascal/artifacts/p1_runs",
        help="Directory for raw JSONL / per-condition summaries (gitignored).",
    )
    parser.add_argument(
        "--compact-out",
        default="companion/pascal/results",
        help="Directory for compact tracked JSON tables (written at end if --finalize).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Dev subset for smoke.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Deterministic smoke: limit=4, still runs all caps.",
    )
    parser.add_argument(
        "--caps",
        default=None,
        help="Comma-separated cap override (default: protocol candidates).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Resume compatible fingerprint rows (default true).",
    )
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    os.environ["PYTHONNOUSERSITE"] = "1"

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    cache = Path(cfg.get("cache_dir", ".hf_cache")).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)
    os.environ.setdefault("HF_HUB_CACHE", str(cache / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HUB_CACHE"])

    if not torch.cuda.is_available():
        print("ERROR: CUDA required; do not run inference on login node", file=sys.stderr)
        return 2

    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    manifest_path = Path(cfg["manifest"])
    manifest = json.loads(manifest_path.read_text())
    vman = verify_manifest_hash(manifest, path=manifest_path)
    if not vman["ok"]:
        print(f"ERROR: manifest hash mismatch: {vman}", file=sys.stderr)
        return 2
    examples = list(manifest["examples"])
    if args.smoke:
        examples = examples[:4]
    elif args.limit is not None:
        examples = examples[: int(args.limit)]
    req_qids = [int(e["question_id"]) for e in examples]
    if len(req_qids) != len(set(req_qids)):
        print("ERROR: duplicate QIDs in requested set", file=sys.stderr)
        return 2

    answers_path = Path(cfg["answers"])
    answers_by_qid: dict[int, list[str]] = {}
    if answers_path.is_file():
        for item in json.loads(answers_path.read_text()).get("answers", []):
            answers_by_qid[int(item["question_id"])] = list(item.get("answers", []))
    else:
        print(f"ERROR: missing answers {answers_path}", file=sys.stderr)
        return 2

    m = int(cfg["original_max_num"])
    min_num = int(cfg.get("min_num", 1))
    if args.caps:
        caps = sorted({int(x) for x in args.caps.split(",")})
    else:
        caps = candidate_max_nums(m, min_num=min_num)
    omitted = omitted_caps(m, min_num=min_num)
    if omitted:
        print(f"NOTE: omitted illegal caps vs min_num={min_num}: {omitted}")
    if len(caps) < 2:
        print(
            f"ERROR: non-informative ablation; only {caps} distinct legal caps",
            file=sys.stderr,
        )
        return 3
    order = execution_order(caps, original_max_num=m)
    print(f"caps={caps} execution_order={order}")

    source_rev = _git_sha()
    print(f"loading model {cfg['model_id']}@{cfg['revision']} source_rev={source_rev}")
    loaded, load_prov = load_internvl_p1(
        cfg["model_id"],
        cfg["revision"],
        prefer_bf16=bool(cfg.get("prefer_bf16", True)),
        use_flash_attn=bool(cfg.get("use_flash_attn", False)),
        max_num=m,
        use_thumbnail=bool(cfg.get("use_thumbnail", True)),
        fallback_dtype_if_no_bf16=str(cfg.get("fallback_dtype_if_no_bf16", "float16")),
    )
    dtype_str = str(loaded.dtype).replace("torch.", "")
    attn = load_prov["attention_backend"]
    print(f"dtype={dtype_str} load_provenance={json.dumps(load_prov)}")
    fin = _finite_check(loaded.model)
    print(f"finite_param_check={fin}")
    if not fin["ok"]:
        print("ERROR: non-finite parameters; stopping", file=sys.stderr)
        return 4

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = int(torch.cuda.get_device_properties(0).total_memory)
    driver = None
    try:
        import subprocess as sp

        driver = sp.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()[0]
    except Exception:
        driver = "unknown"

    warmup_idx = list(cfg.get("warmup_example_indices", [0, 1, 2]))
    # Map indices into full manifest order (not limited list) for stability
    full_examples = list(manifest["examples"])
    warmup_examples = []
    for i in warmup_idx:
        if 0 <= int(i) < len(full_examples):
            warmup_examples.append(full_examples[int(i)])
    if not warmup_examples:
        warmup_examples = [full_examples[0]]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_doc = {
        "task": "pascal_p1_runtime",
        "source_git_revision": source_rev,
        "gpu_name": gpu_name,
        "gpu_total_memory_bytes": gpu_mem,
        "driver_version": driver,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "transformers": __import__("transformers").__version__,
        "dtype": dtype_str,
        "attention_backend": attn,
        "use_flash_attn": bool(cfg.get("use_flash_attn", False)),
        "load_provenance": load_prov,
        "neumann_reference_dtype": "bfloat16",
        "neumann_reference_flash_attn": False,
        "label": "supplementary_development_analysis_post_final_eval",
        "caps": caps,
        "execution_order": order,
        "n_requested": len(req_qids),
        "smoke": bool(args.smoke),
    }
    (out_dir / "runtime.json").write_text(json.dumps(runtime_doc, indent=2) + "\n")

    instruction = str(cfg["instruction"])
    max_new = int(cfg["max_new_tokens"])
    condition_summaries: list[dict[str, Any]] = []

    for cap in order:
        loaded = set_max_num(loaded, cap)
        fp = build_p1_fingerprint(
            repo_root=REPO_ROOT,
            cfg=cfg,
            cfg_path=cfg_path.resolve(),
            max_num=cap,
            manifest_content_sha256=str(vman["canonical_content_sha256"]),
            resolved_dtype=dtype_str,
            attention_backend=attn,
            dtype_policy=str(load_prov.get("resolved_dtype_policy")),
            source_revision=source_rev,
        )
        fp_sha = fingerprint_sha(fp)
        tag = f"max_num_{cap:02d}"
        jsonl_path = out_dir / f"p1_dev_{tag}.jsonl"
        summary_path = out_dir / f"p1_dev_{tag}_summary.json"
        print(f"\n=== condition max_num={cap} fingerprint={fp_sha} ===")

        # Warm-up (excluded)
        print(f"warmup n={len(warmup_examples)} qids={[e['question_id'] for e in warmup_examples]}")
        for we in warmup_examples:
            wv = verify_example_image(we, root=REPO_ROOT)
            if not wv["ok"]:
                print(f"ERROR: warmup image verify failed: {wv}", file=sys.stderr)
                return 2
            _ = run_one_request(
                loaded,
                image_path=str(REPO_ROOT / we["image_relpath"]),
                question=we["question"],
                max_new_tokens=max_new,
                instruction=instruction,
            )
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            assert_output_fingerprint_coherent(jsonl_path, fp_sha)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

        done, n_compat, n_incompat = load_compatible_done_ids(jsonl_path, fp_sha)
        print(
            f"resume compatible_ok={n_compat} incompatible_ignored={n_incompat} "
            f"done_ids={len(done)}"
        )

        selected: dict[int, dict] = {}
        if jsonl_path.is_file():
            with jsonl_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if (rec.get("run_fingerprint") or {}).get(
                        "run_fingerprint_sha256"
                    ) != fp_sha:
                        continue
                    qid = int(rec.get("question_id", -1))
                    if qid in set(req_qids):
                        selected[qid] = rec

        n_retries = 0
        with jsonl_path.open("a") as fout:
            for ex in examples:
                qid = int(ex["question_id"])
                if qid in done:
                    print(f"skip existing compatible qid={qid}")
                    continue
                img_v = verify_example_image(ex, root=REPO_ROOT)
                if not img_v["ok"]:
                    print(f"ERROR: image verify failed qid={qid}: {img_v}", file=sys.stderr)
                    return 2

                status = "ok"
                err = None
                out: dict[str, Any] = {}
                for attempt in range(2):
                    try:
                        out = run_one_request(
                            loaded,
                            image_path=str(REPO_ROOT / ex["image_relpath"]),
                            question=ex["question"],
                            max_new_tokens=max_new,
                            instruction=instruction,
                        )
                        pred = out.get("prediction")
                        if pred is None or (
                            isinstance(pred, float) and not math.isfinite(pred)
                        ):
                            raise RuntimeError("non-finite or null prediction")
                        # basic decode sanity
                        if not isinstance(pred, str):
                            raise RuntimeError(f"prediction type {type(pred)}")
                        status = "ok"
                        err = None
                        if attempt > 0:
                            n_retries += 1
                        break
                    except RuntimeError as exc:
                        # Retry only infrastructure-ish CUDA errors
                        msg = str(exc)
                        retriable = any(
                            k in msg.lower()
                            for k in ("cuda", "cublas", "cudnn", "out of memory")
                        )
                        status = "error"
                        err = msg
                        out = {"prediction": None, "error": msg}
                        if retriable and attempt == 0:
                            print(f"retry qid={qid} after: {msg}")
                            torch.cuda.empty_cache()
                            time.sleep(1.0)
                            continue
                        break

                refs = answers_by_qid.get(qid, [])
                scores = score_example(
                    out.get("prediction"),
                    refs,
                    status=status,
                )
                record = {
                    "question_id": qid,
                    "doc_id": ex.get("doc_id"),
                    "ucsf_document_id": ex.get("ucsf_document_id"),
                    "question": ex["question"],
                    "image_relpath": ex["image_relpath"],
                    "image_sha256_png": ex.get("image_sha256_png"),
                    "image_content_verified_sha256": img_v["sha256"],
                    "model_id": cfg["model_id"],
                    "model_revision": cfg["revision"],
                    "max_num_cap": cap,
                    "actual_tile_count": out.get("actual_tile_count"),
                    "original_size_wh": out.get("original_size_wh"),
                    "thumbnail_appended": out.get("thumbnail_appended"),
                    "image_info": out.get("image_info"),
                    "prediction": out.get("prediction"),
                    "prediction_raw": out.get("prediction_raw"),
                    "status": status,
                    "error": err,
                    "generated_token_length": out.get("generated_token_length"),
                    "generation_cap_reached": out.get("generation_cap_reached"),
                    "generation_seconds": out.get("generation_seconds"),
                    "request_seconds": out.get("request_seconds"),
                    "timing_scope_generation": out.get("timing_scope"),
                    "timing_scope_request": out.get("request_timing_scope"),
                    "gpu_memory_generation": out.get("gpu_memory"),
                    "gpu_memory_request": out.get("request_gpu_memory"),
                    "dtype": dtype_str,
                    "attention_backend": attn,
                    "scores": scores,
                    "primary_anls_strict_v2": scores.get("anls_normalized_strict_v2"),
                    "primary_em": scores.get("exact_match"),
                    "run_fingerprint": fp,
                    "hardware": {
                        "gpu_name": gpu_name,
                        "gpu_total_memory_bytes": gpu_mem,
                        "driver_version": driver,
                    },
                    "software": fp.get("software"),
                    "experiment": "pascal_p1_visual_budget",
                    "label": "supplementary_development_analysis_post_final_eval",
                    "source_git_revision": source_rev,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                selected[qid] = record
                done.add(qid)
                print(
                    f"qid={qid} cap={cap} tiles={record['actual_tile_count']} "
                    f"status={status} anls={record['primary_anls_strict_v2']} "
                    f"gen_s={record.get('generation_seconds')}"
                )

        # Aggregate summary for this condition
        per_example = []
        gen_lat = []
        req_lat = []
        tiles = []
        gen_lens = []
        cap_hits = 0
        peak_alloc = 0
        peak_reserved = 0
        n_ok = n_failed = 0
        for qid in req_qids:
            rec = selected.get(qid)
            if rec is None:
                continue
            if rec.get("status") == "ok":
                n_ok += 1
            else:
                n_failed += 1
            per_example.append(
                {
                    "question_id": qid,
                    "status": rec.get("status"),
                    "prediction": rec.get("prediction"),
                    "anls_normalized_strict_v2": rec.get("primary_anls_strict_v2"),
                    "exact_match": rec.get("primary_em"),
                }
            )
            if rec.get("generation_seconds") is not None:
                gen_lat.append(float(rec["generation_seconds"]))
            if rec.get("request_seconds") is not None:
                req_lat.append(float(rec["request_seconds"]))
            if rec.get("actual_tile_count") is not None:
                tiles.append(int(rec["actual_tile_count"]))
            if rec.get("generated_token_length") is not None:
                gen_lens.append(int(rec["generated_token_length"]))
            if rec.get("generation_cap_reached"):
                cap_hits += 1
            gm = rec.get("gpu_memory_request") or {}
            if gm.get("peak_allocated_bytes") is not None:
                peak_alloc = max(peak_alloc, int(gm["peak_allocated_bytes"]))
            if gm.get("peak_reserved_bytes") is not None:
                peak_reserved = max(peak_reserved, int(gm["peak_reserved_bytes"]))

        n_missing = len(req_qids) - len(selected)
        # Build score rows for summarize_scores
        score_rows = []
        for qid in req_qids:
            rec = selected.get(qid)
            if rec is None:
                continue
            metrics = rec.get("scores")
            if not isinstance(metrics, dict):
                metrics = score_example(
                    rec.get("prediction"),
                    answers_by_qid.get(qid, []),
                    status=str(rec.get("status", "error")),
                )
            score_rows.append(
                {
                    "question_id": qid,
                    "status": rec.get("status", "missing"),
                    "prediction": rec.get("prediction"),
                    "metrics": metrics,
                }
            )
        agg = aggregate_scores(
            score_rows,
            n_requested=len(req_qids),
            required_question_ids=req_qids,
        )

        def _pct(vals: list[float], q: float) -> float | None:
            if not vals:
                return None
            s = sorted(vals)
            if len(s) == 1:
                return float(s[0])
            idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
            return float(s[idx])

        from collections import Counter

        tile_dist = dict(sorted(Counter(tiles).items(), key=lambda x: x[0]))
        summary = {
            "task": "pascal_p1_condition_summary",
            "label": "supplementary_development_analysis_post_final_eval",
            "max_num_cap": cap,
            "n_requested": len(req_qids),
            "n_ok": n_ok,
            "n_failed": n_failed,
            "n_missing": n_missing,
            "primary_anls_strict_v2": agg["mean_anls_requested"],
            "primary_em": agg["mean_exact_match_requested"],
            "scores": agg,
            "tile_count_distribution": tile_dist,
            "tile_count_mean": (sum(tiles) / len(tiles)) if tiles else None,
            "generation_cap_hits": cap_hits,
            "generated_token_length_mean": (
                sum(gen_lens) / len(gen_lens) if gen_lens else None
            ),
            "generated_token_length_median": _pct([float(x) for x in gen_lens], 0.5),
            "latency_generation_only_median_s": _pct(gen_lat, 0.5),
            "latency_generation_only_p95_s": _pct(gen_lat, 0.95),
            "latency_request_median_s": _pct(req_lat, 0.5),
            "latency_request_p95_s": _pct(req_lat, 0.95),
            "latency_warmup_excluded": True,
            "peak_allocated_bytes_request_max_over_examples": peak_alloc,
            "peak_reserved_bytes_request_max_over_examples": peak_reserved,
            "gpu_memory_scope": (
                "Max over per-example request-scoped peaks (image open through "
                "decode); reserved reflects allocator caching, not a minimum VRAM floor."
            ),
            "run_fingerprint_sha256": fp_sha,
            "dtype": dtype_str,
            "attention_backend": attn,
            "gpu_name": gpu_name,
            "source_git_revision": source_rev,
            "jsonl_path": str(jsonl_path.as_posix()),
            "infrastructure_retries": n_retries,
            "manifest_content_sha256": vman["canonical_content_sha256"],
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        condition_summaries.append(summary)
        print(
            f"SUMMARY cap={cap} anls={summary['primary_anls_strict_v2']:.4f} "
            f"em={summary['primary_em']:.4f} med_req={summary['latency_request_median_s']}"
        )

    index = {
        "task": "pascal_p1_run_index",
        "source_git_revision": source_rev,
        "conditions": condition_summaries,
        "runtime": runtime_doc,
        "requested_qids": req_qids,
    }
    (out_dir / "run_index.json").write_text(json.dumps(index, indent=2) + "\n")
    print("Wrote", out_dir / "run_index.json")
    return 0


if __name__ == "__main__":
    # typing Any for _finite_check
    from typing import Any

    raise SystemExit(main())
