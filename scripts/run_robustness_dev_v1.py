#!/usr/bin/env python3
"""Phase 3B bounded development robustness eval (InternVL base + LoRA u100).

Runs three independent image conditions (clean / half_detail / jpeg_q40) for
each model on the fixed robustness subset. Asserts tile counts match across
conditions per image. Does not mutate on-disk source images.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


CONDITIONS = ("clean", "half_detail", "jpeg_q40")


def main() -> int:
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        print("ERROR: export PYTHONNOUSERSITE=1 before python", file=sys.stderr)
        return 2

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/models/internvl3_1b.yaml")
    ap.add_argument(
        "--manifest", default="data/manifests/docvqa_dev_robustness_v1.json"
    )
    ap.add_argument("--answers", default="data/cache/docvqa_dev_v1_answers.json")
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME", ".hf_cache"))
    ap.add_argument(
        "--out-dir", default="results/robustness_v1", help="Directory for jsonl/summaries"
    )
    ap.add_argument(
        "--models",
        default="base,lora100",
        help="Comma list: base, lora100, and/or lora200",
    )
    ap.add_argument(
        "--lora-checkpoint-root",
        default="checkpoints/internvl3_1b_lora_v1",
        help="Root containing update_0100 / update_0200 adapter dirs",
    )
    ap.add_argument(
        "--stem-prefix",
        default=None,
        help="Optional filename stem prefix (default: internvl3_1b or from root)",
    )
    args = ap.parse_args()

    root = _repo_root()
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))

    import torch
    import yaml
    from PIL import Image

    from vlm_doc.adapters.internvl import (
        load_internvl,
        processor_settings_dict,
        run_one_internvl,
    )
    from vlm_doc.image_perturbations import apply_perturbation
    from vlm_doc.metadata import collect_run_metadata
    from vlm_doc.metrics import (
        SCORER_VERSION_PRIMARY,
        aggregate_scores,
        score_example,
        scorer_source_hash,
        validate_answers_sidecar,
    )
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
    from vlm_doc.train.lora import load_language_lora

    if not torch.cuda.is_available():
        print("ERROR: CUDA required", file=sys.stderr)
        return 2

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cache_root = Path(args.cache_dir).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root)
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hub"))

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    vman = verify_manifest_hash(manifest, path=manifest_path)
    if not vman["ok"]:
        print("ERROR: manifest hash mismatch", vman, file=sys.stderr)
        return 2
    examples = list(manifest["examples"])
    req_qids = [int(e["question_id"]) for e in examples]
    n_requested = len(examples)

    answers_blob = json.loads(Path(args.answers).read_text())
    vans = validate_answers_sidecar(
        answers_blob.get("answers", answers_blob), required_question_ids=req_qids
    )
    if not vans["ok"]:
        print("ERROR: answers validation failed", vans, file=sys.stderr)
        return 2
    answers_by_qid = {int(k): v for k, v in vans["answers_by_qid"].items()}
    answer_sha = sha256_file(Path(args.answers))

    instruction = str(cfg.get("instruction"))
    max_new_tokens = int(cfg.get("max_new_tokens", 64))
    prefer_bf16 = bool(cfg.get("prefer_bf16", True))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_root = Path(args.lora_checkpoint_root)
    model_specs = []
    for token in [t.strip() for t in args.models.split(",") if t.strip()]:
        if token == "base":
            model_specs.append(
                {
                    "tag": "base",
                    "adapter_dir": None,
                    "train_update": None,
                    "stem": "internvl3_1b_base",
                }
            )
        elif token in ("lora100", "lora200"):
            step = 100 if token == "lora100" else 200
            adapter_dir = str((ckpt_root / f"update_{step:04d}").as_posix())
            if args.stem_prefix:
                stem = f"{args.stem_prefix}_{token}"
            elif "v2" in str(ckpt_root):
                stem = f"internvl3_1b_lora_v2_u{step}"
            else:
                stem = f"internvl3_1b_lora_u{step}"
            model_specs.append(
                {
                    "tag": token,
                    "adapter_dir": adapter_dir,
                    "train_update": step,
                    "stem": stem,
                }
            )
        else:
            print(f"ERROR: unknown model token {token}", file=sys.stderr)
            return 2

    software_base = {
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "transformers": __import__("transformers").__version__,
        "python": sys.version.split()[0],
        "pillow": Image.__version__,
    }
    run_meta = collect_run_metadata(root)
    gpu_name = torch.cuda.get_device_name(0)
    gpu_total = int(torch.cuda.get_device_properties(0).total_memory)

    tile_assert_failures: list[dict] = []
    run_index: list[dict] = []

    for spec in model_specs:
        print(f"=== LOAD MODEL {spec['tag']} ===")
        loaded = load_internvl(
            cfg["model_id"],
            cfg["revision"],
            prefer_bf16=prefer_bf16,
            use_flash_attn=bool(cfg.get("use_flash_attn", False)),
            max_num=int(cfg.get("max_num", 12)),
            use_thumbnail=bool(cfg.get("use_thumbnail", True)),
        )
        adapter_config_sha = None
        adapter_weight_shas = None
        if spec["adapter_dir"]:
            adapter_path = Path(spec["adapter_dir"])
            if not adapter_path.is_dir():
                print(f"ERROR: missing adapter {adapter_path}", file=sys.stderr)
                return 2
            load_language_lora(loaded.model, str(adapter_path))
            adapter_config_sha = sha256_file(adapter_path / "adapter_config.json")
            adapter_weight_shas = {
                p.name: sha256_file(p)
                for p in sorted(adapter_path.glob("*.safetensors"))
            }
            print(f"loaded_lora {adapter_path} {adapter_weight_shas}")

        proc_settings = processor_settings_dict(loaded)
        inference_sources = [
            "src/vlm_doc/adapters/internvl.py",
            "src/vlm_doc/adapters/internvl_conv.py",
            "src/vlm_doc/image_perturbations.py",
            "scripts/run_robustness_dev_v1.py",
            str(cfg_path.as_posix()),
        ]
        if spec["adapter_dir"]:
            inference_sources.append("src/vlm_doc/train/lora.py")
        source_hashes = hash_paths(inference_sources)
        dtype_str = str(loaded.dtype).replace("torch.", "")

        # Warmup on clean first example
        warm = examples[0]
        img_v = verify_example_image(warm, root=root)
        if not img_v["ok"]:
            print("ERROR warmup image", img_v, file=sys.stderr)
            return 2
        warm_img, _ = apply_perturbation(
            Image.open(warm["image_relpath"]).convert("RGB"), perturbation="clean"
        )
        _ = run_one_internvl(
            loaded,
            image=warm_img,
            question=warm["question"],
            max_new_tokens=max_new_tokens,
            instruction=instruction,
        )
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Per-condition fingerprints and outputs
        condition_state: dict[str, dict] = {}
        for cond in CONDITIONS:
            pert_fp = {
                "perturbation_id": cond,
                "settings_schema": "phase03b_v1",
                "half_detail_resample": "BICUBIC" if cond == "half_detail" else None,
                "jpeg_quality": 40 if cond == "jpeg_q40" else None,
                "jpeg_subsampling": 2 if cond == "jpeg_q40" else None,
            }
            fingerprint = build_inference_fingerprint(
                model_id=cfg["model_id"],
                model_revision=loaded.revision,
                manifest_content_sha256=vman["canonical_content_sha256"],
                instruction=instruction,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                prefer_bf16=prefer_bf16,
                seed=seed,
                config_path=str(cfg_path.as_posix()),
                config_sha256=sha256_file(cfg_path),
                resolved_dtype=dtype_str,
                processor_settings=proc_settings,
                software=software_base,
                inference_source_hashes=source_hashes,
                adapter_dir=spec["adapter_dir"],
                adapter_config_sha256=adapter_config_sha,
                adapter_weights_sha256=adapter_weight_shas,
                train_update=spec["train_update"],
                perturbation=pert_fp,
            )
            out_jsonl = out_dir / f"{spec['stem']}__{cond}.jsonl"
            summary_json = out_dir / f"{spec['stem']}__{cond}_summary.json"
            try:
                assert_output_fingerprint_coherent(
                    out_jsonl, fingerprint["run_fingerprint_sha256"]
                )
            except RuntimeError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 2
            done, n_ok, n_bad = load_compatible_done_ids(
                out_jsonl, fingerprint["run_fingerprint_sha256"]
            )
            print(
                f"{spec['tag']}/{cond} resume done={len(done)} ok={n_ok} incompat={n_bad} "
                f"fp={fingerprint['run_fingerprint_sha256'][:12]}"
            )
            condition_state[cond] = {
                "fingerprint": fingerprint,
                "out_jsonl": out_jsonl,
                "summary_json": summary_json,
                "done": done,
                "selected": {},
            }
            if out_jsonl.is_file():
                for line in out_jsonl.read_text().splitlines():
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    fp = (rec.get("run_fingerprint") or {}).get("run_fingerprint_sha256")
                    if fp != fingerprint["run_fingerprint_sha256"]:
                        continue
                    condition_state[cond]["selected"][int(rec["question_id"])] = rec

        # Evaluate each example across all conditions (assert tiles)
        for ex in examples:
            qid = int(ex["question_id"])
            img_v = verify_example_image(ex, root=root)
            if not img_v["ok"]:
                print(f"ERROR image verify qid={qid}: {img_v}", file=sys.stderr)
                return 2
            clean_rgb = Image.open(ex["image_relpath"]).convert("RGB")
            tile_by_cond: dict[str, int] = {}
            for cond in CONDITIONS:
                st = condition_state[cond]
                if qid in st["done"] and qid in st["selected"]:
                    prev = st["selected"][qid]
                    tile_by_cond[cond] = int(
                        (prev.get("image_info") or {}).get("tile_count")
                        or (prev.get("perturbation") or {}).get("observed_tile_count")
                        or -1
                    )
                    continue

                image_t, pert_meta = apply_perturbation(clean_rgb, perturbation=cond)
                record = {
                    "question_id": qid,
                    "doc_id": ex.get("doc_id"),
                    "ucsf_document_id": ex.get("ucsf_document_id"),
                    "ucsf_document_page_no": ex.get("ucsf_document_page_no"),
                    "question": ex["question"],
                    "image_relpath": ex["image_relpath"],
                    "image_sha256_png": ex.get("image_sha256_png"),
                    "image_content_verified_sha256": img_v["sha256"],
                    "model_id": loaded.model_id,
                    "model_revision": loaded.revision,
                    "processor_settings": proc_settings,
                    "manifest_name": manifest.get("name"),
                    "manifest_content_sha256": vman["canonical_content_sha256"],
                    "manifest_raw_file_sha256": vman["raw_file_sha256"],
                    "run_fingerprint": st["fingerprint"],
                    "run_metadata": run_meta,
                    "software": {
                        **software_base,
                        "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE"),
                    },
                    "hardware": {
                        "gpu_name": gpu_name,
                        "gpu_total_memory_bytes": gpu_total,
                    },
                    "seed": seed,
                    "total_unique_parameters": loaded.total_parameters,
                    "perturbation": pert_meta,
                    "status": "ok",
                    "error": None,
                }
                try:
                    out = run_one_internvl(
                        loaded,
                        image=image_t,
                        question=ex["question"],
                        max_new_tokens=max_new_tokens,
                        instruction=instruction,
                    )
                    record.update(out)
                    tile_by_cond[cond] = int(out["image_info"]["tile_count"])
                    record["perturbation"]["observed_tile_count"] = tile_by_cond[cond]
                    refs = answers_by_qid[qid]
                    record["reference_answers"] = refs
                    record["metrics"] = score_example(
                        out["prediction"],
                        refs,
                        status="ok",
                        primary=SCORER_VERSION_PRIMARY,
                    )
                    record["scoring_provenance"] = build_scoring_provenance(
                        primary_metric=SCORER_VERSION_PRIMARY,
                        answer_sidecar_sha256=answer_sha,
                        scorer_source_sha256=scorer_source_hash(),
                    )
                    print(
                        f"ok {spec['tag']}/{cond} qid={qid} tiles={tile_by_cond[cond]} "
                        f"pred={out['prediction']!r}"
                    )
                except Exception as exc:  # noqa: BLE001
                    record["status"] = "error"
                    record["error"] = repr(exc)
                    record["metrics"] = score_example(
                        None, answers_by_qid[qid], status="error", primary=SCORER_VERSION_PRIMARY
                    )
                    tile_by_cond[cond] = -1
                    print(f"ERR {spec['tag']}/{cond} qid={qid}: {exc}", file=sys.stderr)

                with st["out_jsonl"].open("a") as fout:
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                st["selected"][qid] = record
                st["done"].add(qid)

            tiles = [tile_by_cond[c] for c in CONDITIONS]
            if len(set(tiles)) != 1 or tiles[0] < 1:
                tile_assert_failures.append(
                    {"model": spec["tag"], "question_id": qid, "tiles": tile_by_cond}
                )
                print(
                    f"ERROR tile mismatch model={spec['tag']} qid={qid} {tile_by_cond}",
                    file=sys.stderr,
                )
                return 2

        # Write summaries per condition
        for cond in CONDITIONS:
            st = condition_state[cond]
            selected = [st["selected"][q] for q in req_qids if q in st["selected"]]
            by_qid = {int(r["question_id"]): r for r in selected}
            scored = []
            for q in req_qids:
                rec = by_qid.get(q)
                if rec is None:
                    continue
                m = rec.get("metrics") or score_example(
                    None, answers_by_qid[q], status="error", primary=SCORER_VERSION_PRIMARY
                )
                scored.append(
                    {
                        "question_id": q,
                        "status": rec.get("status"),
                        "prediction": rec.get("prediction"),
                        "reference_answers": answers_by_qid[q],
                        "metrics": m,
                        "generation_cap_reached": bool(rec.get("generation_cap_reached")),
                    }
                )
            agg = aggregate_scores(
                scored,
                n_requested=n_requested,
                required_question_ids=req_qids,
                primary_key=SCORER_VERSION_PRIMARY,
            )
            n_cap = sum(1 for r in scored if r.get("generation_cap_reached"))
            summary = {
                "task": "docvqa_robustness_v1",
                "model_tag": spec["tag"],
                "perturbation_id": cond,
                "manifest_name": manifest.get("name"),
                "manifest_content_sha256": vman["canonical_content_sha256"],
                "manifest_raw_file_sha256": vman["raw_file_sha256"],
                "n_requested": n_requested,
                "n_ok": int(agg["n_completed_ok"]),
                "n_failed": int(agg["n_failed"]),
                "n_missing": int(agg["n_missing"]),
                "n_generation_cap_hits": n_cap,
                "primary_metric": SCORER_VERSION_PRIMARY,
                "anls_normalized_strict_v2": agg["mean_anls_requested"],
                "exact_match": agg["mean_exact_match_requested"],
                "aggregate": agg,
                "run_fingerprint": st["fingerprint"],
                "adapter_dir": spec["adapter_dir"],
                "train_update": spec["train_update"],
                "hardware": {"gpu_name": gpu_name, "gpu_total_memory_bytes": gpu_total},
                "software": software_base,
            }
            st["summary_json"].write_text(json.dumps(summary, indent=2) + "\n")
            run_index.append(
                {
                    "model_tag": spec["tag"],
                    "perturbation_id": cond,
                    "jsonl": str(st["out_jsonl"].as_posix()),
                    "summary": str(st["summary_json"].as_posix()),
                    "anls": summary["anls_normalized_strict_v2"],
                    "em": summary["exact_match"],
                    "n_ok": summary["n_ok"],
                    "n_failed": summary["n_failed"],
                    "n_missing": summary["n_missing"],
                    "n_generation_cap_hits": n_cap,
                }
            )
            print(
                f"SUMMARY {spec['tag']}/{cond} ANLS={summary['anls_normalized_strict_v2']:.4f} "
                f"EM={summary['exact_match']:.4f}"
            )

        # Free model before next
        del loaded
        torch.cuda.empty_cache()

    index_path = out_dir / "run_index.json"
    index_path.write_text(
        json.dumps(
            {
                "task": "robustness_v1_run_index",
                "manifest": str(manifest_path.as_posix()),
                "manifest_content_sha256": vman["canonical_content_sha256"],
                "tile_assert_failures": tile_assert_failures,
                "runs": run_index,
                "timestamp_unix": time.time(),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"WROTE {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
