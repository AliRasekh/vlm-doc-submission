#!/usr/bin/env python3
"""General evaluation entry point for frozen DocVQA manifests (SmolVLM adapter)."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def _bytes_to_gib(n: int) -> float:
    return float(n) / (1024.0**3)


def _processor_settings(processor) -> dict:
    """Record native processor / image-processor settings (no mutation)."""
    out: dict = {}
    ip = getattr(processor, "image_processor", None)
    if ip is not None:
        for key in (
            "do_image_splitting",
            "do_resize",
            "size",
            "max_image_size",
            "resample",
            "image_mean",
            "image_std",
            "do_normalize",
            "do_convert_rgb",
        ):
            if hasattr(ip, key):
                val = getattr(ip, key)
                try:
                    json.dumps(val)
                    out[key] = val
                except TypeError:
                    out[key] = str(val)
        out["image_processor_class"] = type(ip).__name__
    out["processor_class"] = type(processor).__name__
    if hasattr(processor, "chat_template"):
        out["has_chat_template"] = processor.chat_template is not None
    return out


def main() -> int:
    # Fail-fast: PYTHONNOUSERSITE must already be set by the launcher.
    # Setting it here does NOT unload user-site modules already imported.
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        print(
            "ERROR: PYTHONNOUSERSITE=1 must be exported by the shell/Slurm "
            "launcher before invoking Python. Setting it inside main() cannot "
            "retroactively disable user-site imports.",
            file=sys.stderr,
        )
        return 2

    parser = argparse.ArgumentParser(description="vlm_doc DocVQA evaluation runner")
    parser.add_argument("--config", default="configs/models/smolvlm_256m.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--answers",
        default=None,
        help="Local answers sidecar JSON (not fed to the model). "
        "Default: data/cache/<manifest_stem>_answers.json",
    )
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HOME", ".hf_cache"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--score",
        action="store_true",
        help="Attach ANLS/EM using local answers sidecar (primary=strict_v2).",
    )
    parser.add_argument(
        "--allow-mixed-hardware-resume",
        action="store_true",
        help="Permit resume across different GPU names; summary will label mixed HW.",
    )
    parser.add_argument(
        "--lora-adapter",
        default=None,
        help="Optional PEFT adapter directory (InternVL language_model LoRA).",
    )
    parser.add_argument(
        "--train-update",
        type=int,
        default=None,
        help="Training update index for fingerprinting LoRA checkpoints (e.g. 100/200).",
    )
    args = parser.parse_args()

    root = _repo_root()
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))

    cache_root = Path(args.cache_dir).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root)
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hub"))
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))

    import torch
    import yaml
    from PIL import Image

    from vlm_doc.infer import run_one
    from vlm_doc.metadata import collect_run_metadata
    from vlm_doc.metrics import (
        SCORER_VERSION_PRIMARY,
        aggregate_scores,
        score_example,
        scorer_source_hash,
        validate_answers_sidecar,
    )
    from vlm_doc.model import load_vlm
    from vlm_doc.run_fingerprint import (
        assert_output_fingerprint_coherent,
        build_inference_fingerprint,
        build_scoring_provenance,
        hash_paths,
        load_compatible_done_ids,
        sha256_file,
        verify_example_image,
        verify_manifest_hash,
    )

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    vman = verify_manifest_hash(manifest, path=manifest_path)
    if not vman["ok"]:
        print(
            f"ERROR: manifest canonical content hash mismatch "
            f"stored={vman['stored_manifest_sha256']} "
            f"canonical={vman['canonical_content_sha256']} "
            f"(raw_file_sha256={vman['raw_file_sha256']}). "
            f"canonical excludes self-hash field; raw file SHA may differ.",
            file=sys.stderr,
        )
        return 2
    manifest_content_sha = vman["canonical_content_sha256"]
    examples = list(manifest["examples"])
    if args.limit is not None:
        examples = examples[: args.limit]
    n_requested = len(examples)
    req_qids = [int(e["question_id"]) for e in examples]

    answers_path = Path(args.answers) if args.answers else Path(
        f"data/cache/{manifest_path.stem}_answers.json"
    )
    answers_by_qid: dict[int, list[str]] = {}
    answer_sha = None
    if answers_path.is_file():
        answers_blob = json.loads(answers_path.read_text())
        answers_list = answers_blob.get("answers", answers_blob)
        if args.score:
            vans = validate_answers_sidecar(answers_list, required_question_ids=req_qids)
            if not vans["ok"]:
                print(
                    "ERROR: answer sidecar validation failed: "
                    + json.dumps({k: vans[k] for k in vans if k != "answers_by_qid"}),
                    file=sys.stderr,
                )
                return 2
            answers_by_qid = {int(k): v for k, v in vans["answers_by_qid"].items()}
        else:
            for item in answers_list:
                answers_by_qid[int(item["question_id"])] = list(item.get("answers", []))
        answer_sha = sha256_file(answers_path)
    elif args.score:
        print(f"ERROR: --score requires answers file at {answers_path}", file=sys.stderr)
        return 2

    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for evaluation jobs", file=sys.stderr)
        return 2

    instruction = str(cfg.get("instruction"))
    max_new_tokens = int(cfg.get("max_new_tokens", 64))
    prefer_bf16 = bool(cfg.get("prefer_bf16", True))
    adapter_name = str(cfg.get("adapter", "smolvlm")).lower()

    print(f"loading adapter={adapter_name} {cfg['model_id']}@{cfg['revision']}")
    if adapter_name in ("internvl", "internvl3"):
        from vlm_doc.adapters.internvl import (
            load_internvl,
            processor_settings_dict as internvl_processor_settings,
            run_one_internvl,
        )

        loaded = load_internvl(
            cfg["model_id"],
            cfg["revision"],
            prefer_bf16=prefer_bf16,
            use_flash_attn=bool(cfg.get("use_flash_attn", False)),
            max_num=int(cfg.get("max_num", 12)),
            use_thumbnail=bool(cfg.get("use_thumbnail", True)),
        )
        adapter_dir = args.lora_adapter
        adapter_config_sha = None
        adapter_weight_shas = None
        if adapter_dir:
            from vlm_doc.train.lora import load_language_lora

            adapter_path = Path(adapter_dir)
            if not adapter_path.is_dir():
                print(f"ERROR: missing LoRA adapter dir {adapter_path}", file=sys.stderr)
                return 2
            load_language_lora(loaded.model, str(adapter_path))
            cfg_json = adapter_path / "adapter_config.json"
            adapter_config_sha = (
                sha256_file(cfg_json) if cfg_json.is_file() else None
            )
            adapter_weight_shas = {}
            for p in sorted(adapter_path.glob("*.safetensors")) + sorted(
                adapter_path.glob("adapter_model.bin")
            ):
                adapter_weight_shas[p.name] = sha256_file(p)
            print(
                f"loaded_lora_adapter={adapter_path} "
                f"weights={adapter_weight_shas}"
            )
        run_fn = run_one_internvl
        proc_settings = internvl_processor_settings(loaded)
        inference_sources = [
            "src/vlm_doc/adapters/internvl.py",
            "src/vlm_doc/adapters/internvl_conv.py",
            "src/vlm_doc/model.py",
            "scripts/run_eval.py",
            str(cfg_path.as_posix()),
        ]
        if adapter_dir:
            inference_sources.append("src/vlm_doc/train/lora.py")
            inference_sources.append("src/vlm_doc/train/collator.py")
        model_revision_field = loaded.revision
        processor_revision_field = loaded.revision
        total_params = loaded.total_parameters
        dtype_str = str(loaded.dtype).replace("torch.", "")
        bf16_flag = loaded.bf16_supported
        model_id_field = loaded.model_id
        lora_adapter_dir = str(Path(adapter_dir).as_posix()) if adapter_dir else None
        lora_adapter_config_sha = adapter_config_sha
        lora_adapter_weight_shas = adapter_weight_shas
        lora_train_update = args.train_update
    else:
        if args.lora_adapter:
            print("ERROR: --lora-adapter only supported for InternVL adapter", file=sys.stderr)
            return 2
        loaded = load_vlm(
            cfg["model_id"],
            cfg["revision"],
            prefer_bf16=prefer_bf16,
        )
        run_fn = run_one
        proc_settings = _processor_settings(loaded.processor)
        inference_sources = [
            "src/vlm_doc/infer.py",
            "src/vlm_doc/model.py",
            "scripts/run_eval.py",
            str(cfg_path.as_posix()),
        ]
        model_revision_field = loaded.revision
        processor_revision_field = loaded.revision
        total_params = loaded.total_parameters
        dtype_str = str(loaded.dtype).replace("torch.", "")
        bf16_flag = loaded.bf16_supported
        model_id_field = loaded.model_id
        lora_adapter_dir = None
        lora_adapter_config_sha = None
        lora_adapter_weight_shas = None
        lora_train_update = None

    print(
        f"total_unique_parameters={total_params} "
        f"dtype={dtype_str} bf16_supported={bf16_flag}"
    )
    if total_params >= 1_000_000_000:
        print("ERROR: parameter count exceeds 1e9 budget", file=sys.stderr)
        return 2

    print(f"processor_settings={json.dumps(proc_settings, default=str)}")

    source_hashes = hash_paths(inference_sources)
    software = {
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "transformers": __import__("transformers").__version__,
        "python": sys.version.split()[0],
    }
    fingerprint = build_inference_fingerprint(
        model_id=cfg["model_id"],
        model_revision=cfg["revision"],
        manifest_content_sha256=manifest_content_sha,
        instruction=instruction,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        prefer_bf16=prefer_bf16,
        seed=seed,
        config_path=str(cfg_path.as_posix()),
        config_sha256=sha256_file(cfg_path),
        resolved_dtype=dtype_str,
        processor_settings=proc_settings,
        software=software,
        inference_source_hashes=source_hashes,
        adapter_dir=lora_adapter_dir,
        adapter_config_sha256=lora_adapter_config_sha,
        adapter_weights_sha256=lora_adapter_weight_shas,
        train_update=lora_train_update,
    )
    print(f"run_fingerprint={fingerprint['run_fingerprint_sha256']}")

    gpu_name = torch.cuda.get_device_name(0)
    gpu_total = int(torch.cuda.get_device_properties(0).total_memory)
    run_meta = collect_run_metadata(root)

    # Warm-up excluded from latency stats
    warm = examples[0]
    img_check = verify_example_image(warm, root=root)
    if not img_check["ok"]:
        print(f"ERROR: warmup image verification failed: {img_check}", file=sys.stderr)
        return 2
    print(f"warmup qid={warm['question_id']}")
    _ = run_fn(
        loaded,
        image=Image.open(warm["image_relpath"]).convert("RGB"),
        question=warm["question"],
        max_new_tokens=max_new_tokens,
        instruction=instruction,
    )
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        assert_output_fingerprint_coherent(
            out_path, fingerprint["run_fingerprint_sha256"]
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    done, n_compat, n_incompat = load_compatible_done_ids(
        out_path, fingerprint["run_fingerprint_sha256"]
    )
    print(
        f"resume compatible_ok={n_compat} incompatible_ignored={n_incompat} "
        f"done_ids={len(done)}"
    )

    # Deduplicated selected records (last write wins) for scores/latency/memory
    selected: dict[int, dict] = {}
    gpu_names_seen: set[str] = set()

    if out_path.is_file():
        with out_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fp = (rec.get("run_fingerprint") or {}).get("run_fingerprint_sha256")
                if fp != fingerprint["run_fingerprint_sha256"]:
                    continue
                qid = int(rec.get("question_id", -1))
                if qid not in set(req_qids):
                    continue
                selected[qid] = rec
                hw = rec.get("hardware") or {}
                if hw.get("gpu_name"):
                    gpu_names_seen.add(str(hw["gpu_name"]))

    if gpu_names_seen and (gpu_name not in gpu_names_seen) and not args.allow_mixed_hardware_resume:
        print(
            f"ERROR: existing compatible records used GPUs {sorted(gpu_names_seen)} "
            f"but current GPU is {gpu_name}. Use a new --out-jsonl or pass "
            f"--allow-mixed-hardware-resume (summary will not claim single-GPU timings).",
            file=sys.stderr,
        )
        return 2

    with out_path.open("a") as fout:
        for ex in examples:
            qid = int(ex["question_id"])
            if qid in done:
                print(f"skip existing compatible qid={qid}")
                continue

            img_v = verify_example_image(ex, root=root)
            if not img_v["ok"]:
                print(f"ERROR: image verify failed qid={qid}: {img_v}", file=sys.stderr)
                return 2

            record = {
                "question_id": qid,
                "doc_id": ex.get("doc_id"),
                "ucsf_document_id": ex.get("ucsf_document_id"),
                "ucsf_document_page_no": ex.get("ucsf_document_page_no"),
                "question": ex["question"],
                "image_relpath": ex["image_relpath"],
                "image_sha256_png": ex.get("image_sha256_png"),
                "image_content_verified_sha256": img_v["sha256"],
                "model_id": model_id_field,
                "model_revision": model_revision_field,
                "processor_revision": processor_revision_field,
                "processor_settings": proc_settings,
                "manifest_name": manifest.get("name"),
                "manifest_content_sha256": manifest_content_sha,
                "manifest_raw_file_sha256": vman["raw_file_sha256"],
                "run_fingerprint": fingerprint,
                "run_metadata": run_meta,
                "software": {
                    **software,
                    "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE"),
                },
                "hardware": {
                    "gpu_name": gpu_name,
                    "gpu_total_memory_bytes": gpu_total,
                },
                "seed": seed,
                "total_unique_parameters": total_params,
                "status": "ok",
                "error": None,
                "prediction": None,
                "prediction_raw": None,
            }
            t_wall0 = time.perf_counter()
            try:
                image = Image.open(ex["image_relpath"]).convert("RGB")
                out = run_fn(
                    loaded,
                    image=image,
                    question=ex["question"],
                    max_new_tokens=max_new_tokens,
                    instruction=instruction,
                )
                record.update(out)
                refs = answers_by_qid.get(qid)
                if args.score:
                    if refs is None:
                        raise RuntimeError(f"Missing reference answers for qid={qid}")
                    record["reference_answers"] = refs
                    record["metrics"] = score_example(
                        out["prediction"], refs, status="ok", primary=SCORER_VERSION_PRIMARY
                    )
                    record["scoring_provenance"] = build_scoring_provenance(
                        primary_metric=SCORER_VERSION_PRIMARY,
                        answer_sidecar_sha256=answer_sha or "",
                        scorer_source_sha256=scorer_source_hash(),
                    )
                elif refs is not None:
                    record["reference_answers_local"] = refs
                print(
                    f"ok qid={qid} sec={out['generation_seconds']:.3f} "
                    f"pred={out['prediction']!r} "
                    f"gen_len={out.get('generated_token_length')} "
                    f"cap={out.get('generation_cap_reached')}"
                )
            except Exception as exc:  # noqa: BLE001
                record["status"] = "error"
                record["error"] = repr(exc)
                refs = answers_by_qid.get(qid, [])
                if args.score:
                    record["reference_answers"] = refs
                    record["metrics"] = score_example(
                        None, refs, status="error", primary=SCORER_VERSION_PRIMARY
                    )
                print(f"ERROR qid={qid}: {exc}", file=sys.stderr)
            record["wall_seconds"] = time.perf_counter() - t_wall0
            fout.write(json.dumps(record) + "\n")
            fout.flush()
            selected[qid] = record
            gpu_names_seen.add(gpu_name)

    ordered = [selected[qid] for qid in req_qids if qid in selected]

    # Latency / memory from the same deduplicated selected records
    latencies: list[float] = []
    peak_alloc = 0
    peak_reserved = 0
    gen_lens: list[int] = []
    n_cap = 0
    for r in ordered:
        if r.get("status") == "ok" and r.get("generation_seconds") is not None:
            latencies.append(float(r["generation_seconds"]))
        gm = r.get("gpu_memory") or {}
        peak_alloc = max(peak_alloc, int(gm.get("peak_allocated_bytes") or 0))
        peak_reserved = max(peak_reserved, int(gm.get("peak_reserved_bytes") or 0))
        if r.get("generated_token_length") is not None:
            gen_lens.append(int(r["generated_token_length"]))
        if r.get("generation_cap_reached"):
            n_cap += 1

    mixed_hw = len(gpu_names_seen) > 1
    hw_summary = {
        "gpu_names_seen": sorted(gpu_names_seen) or [gpu_name],
        "current_gpu_name": gpu_name,
        "gpu_total_memory_bytes": gpu_total,
        "mixed_hardware": mixed_hw,
        "timing_label": (
            "mixed_hardware_resume_do_not_treat_as_single_gpu_measurement"
            if mixed_hw
            else "single_gpu"
        ),
        "note_gpu_total": "Device capacity from CUDA properties; not MaxRSS.",
    }

    summary = {
        "task": "docvqa_eval",
        "manifest_name": manifest.get("name"),
        "manifest_content_sha256": manifest_content_sha,
        "manifest_raw_file_sha256": vman["raw_file_sha256"],
        "manifest_hash_note": vman["note"],
        "manifest_role": manifest.get("role"),
        "model_id": model_id_field,
        "model_revision": model_revision_field,
        "total_unique_parameters": total_params,
        "dtype": dtype_str,
        "bf16_supported": bf16_flag,
        "adapter": adapter_name,
        "processor_settings": proc_settings,
        "seed": seed,
        "instruction": instruction,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "run_fingerprint": fingerprint,
        "run_metadata": run_meta,
        "hardware": hw_summary,
        "host_memory_note": (
            "Slurm MaxRSS is host (CPU) resident memory for the job, not GPU memory."
        ),
        "gpu_memory_peaks": {
            "peak_allocated_bytes": peak_alloc,
            "peak_reserved_bytes": peak_reserved,
            "peak_allocated_gib": _bytes_to_gib(peak_alloc),
            "peak_reserved_gib": _bytes_to_gib(peak_reserved),
            "gib_conversion": "bytes / 1024**3",
            "scope": (
                "Max over per-example torch.cuda.max_memory_{allocated,reserved} "
                "from the same deduplicated selected records; not nvidia-smi."
            ),
        },
        "latency_seconds": {
            "n": len(latencies),
            "median": statistics.median(latencies) if latencies else None,
            "p95": _percentile(latencies, 95),
            "mean": (sum(latencies) / len(latencies)) if latencies else None,
            "warmup_excluded": True,
            "cuda_synchronized": True,
            "source": "deduplicated_selected_records",
            "hardware_timing_label": hw_summary["timing_label"],
        },
        "generation_diagnostics": {
            "n_with_length": len(gen_lens),
            "median_generated_token_length": (
                statistics.median(gen_lens) if gen_lens else None
            ),
            "n_generation_cap_reached": n_cap,
            "max_new_tokens": max_new_tokens,
        },
        "software": software,
        "prediction_whitespace_note": (
            "prediction_raw is saved before strip(); prediction is stripped. "
            "Older Phase-2A records only stored stripped decoded outputs."
        ),
        "note": (
            "Development/smoke scores are not full-benchmark results and must not "
            "be presented as unbiased final DocVQA performance."
        ),
    }

    if args.score:
        for r in ordered:
            if "metrics" not in r:
                r["metrics"] = score_example(
                    r.get("prediction"),
                    r.get("reference_answers")
                    or answers_by_qid.get(int(r["question_id"]), []),
                    status=r.get("status", "error"),
                    primary=SCORER_VERSION_PRIMARY,
                )
        summary["scoring_provenance"] = build_scoring_provenance(
            primary_metric=SCORER_VERSION_PRIMARY,
            answer_sidecar_sha256=answer_sha or "",
            scorer_source_sha256=scorer_source_hash(),
        )
        summary["scores"] = aggregate_scores(
            ordered,
            n_requested=n_requested,
            required_question_ids=req_qids,
            primary_key=SCORER_VERSION_PRIMARY,
        )
    else:
        summary["counts"] = {
            "n_requested": n_requested,
            "n_records": len(ordered),
            "n_completed_ok": sum(1 for r in ordered if r.get("status") == "ok"),
            "n_failed": sum(1 for r in ordered if r.get("status") != "ok"),
            "n_missing": n_requested - len(ordered),
        }

    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {args.summary_json}")
    if args.score:
        s = summary["scores"]
        print(
            f"ANLS_strict_v2_requested={s['mean_anls_requested']:.4f} "
            f"EM_requested={s['mean_exact_match_requested']:.4f} "
            f"inclusive_v1={s['mean_anls_normalized_inclusive_v1_requested']:.4f} "
            f"ok={s['n_completed_ok']} fail={s['n_failed']} missing={s['n_missing']}"
        )
    n_fail = sum(1 for r in ordered if r.get("status") != "ok") + (
        n_requested - len(ordered)
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
