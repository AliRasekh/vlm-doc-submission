#!/usr/bin/env python3
"""DocVQA smoke inference for SmolVLM (batch size 1, greedy)."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import yaml
from PIL import Image


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/models/smolvlm_256m.yaml",
    )
    parser.add_argument(
        "--manifest",
        default="data/manifests/docvqa_smoke_v1.json",
    )
    parser.add_argument(
        "--answers",
        default="data/cache/docvqa_smoke_v1_answers.json",
        help="Optional local reference answers (not fed to the model).",
    )
    parser.add_argument(
        "--out-jsonl",
        default="results/smoke_smolvlm256m.jsonl",
    )
    parser.add_argument(
        "--summary-json",
        default="results/smoke_smolvlm256m_summary.json",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("HF_HOME", ".hf_cache"),
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    root = _repo_root()
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))

    cache_root = Path(args.cache_dir).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root)
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HUB_CACHE"])
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))

    from vlm_doc.infer import run_one
    from vlm_doc.metadata import collect_run_metadata
    from vlm_doc.model import load_vlm

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    manifest = json.loads(Path(args.manifest).read_text())
    examples = list(manifest["examples"])
    if args.limit is not None:
        examples = examples[: args.limit]

    answers_by_qid: dict[int, list[str]] = {}
    answers_path = Path(args.answers)
    if answers_path.is_file():
        sidecars = json.loads(answers_path.read_text()).get("answers", [])
        for item in sidecars:
            answers_by_qid[int(item["question_id"])] = list(item.get("answers", []))

    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for the smoke job", file=sys.stderr)
        return 2

    print(f"loading {cfg['model_id']}@{cfg['revision']}")
    loaded = load_vlm(
        cfg["model_id"],
        cfg["revision"],
        prefer_bf16=bool(cfg.get("prefer_bf16", True)),
    )
    print(
        f"total_unique_parameters={loaded.total_parameters} "
        f"dtype={loaded.dtype} bf16_supported={loaded.bf16_supported}"
    )

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory
    props = {
        "gpu_name": gpu_name,
        "gpu_total_memory_bytes": int(gpu_mem),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers_version": __import__("transformers").__version__,
        "model_id": loaded.model_id,
        "model_revision": loaded.revision,
        "total_unique_parameters": loaded.total_parameters,
        "bf16_supported": loaded.bf16_supported,
        "dtype": str(loaded.dtype).replace("torch.", ""),
        "seed": seed,
        "run_metadata": collect_run_metadata(root),
        "manifest_name": manifest.get("name"),
        "manifest_dataset_revision": manifest.get("dataset_revision"),
    }

    # Warm-up (not timed as benchmark): first example once, discard metrics
    warm = examples[0]
    warm_img = Image.open(warm["image_relpath"]).convert("RGB")
    print(f"warmup qid={warm['question_id']}")
    _ = run_one(
        loaded,
        image=warm_img,
        question=warm["question"],
        max_new_tokens=int(cfg.get("max_new_tokens", 64)),
        instruction=str(cfg.get("instruction")),
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Resumable: skip completed question_ids
    done: set[int] = set()
    if out_path.is_file():
        with out_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("status") == "ok":
                        done.add(int(rec["question_id"]))
                except json.JSONDecodeError:
                    continue

    results_preview = []
    n_ok = 0
    n_err = 0
    with out_path.open("a") as fout:
        for ex in examples:
            qid = int(ex["question_id"])
            if qid in done:
                print(f"skip existing qid={qid}")
                continue
            record = {
                "question_id": qid,
                "doc_id": ex.get("doc_id"),
                "ucsf_document_id": ex.get("ucsf_document_id"),
                "ucsf_document_page_no": ex.get("ucsf_document_page_no"),
                "question": ex["question"],
                "image_relpath": ex["image_relpath"],
                "model_id": loaded.model_id,
                "model_revision": loaded.revision,
                "status": "ok",
                "error": None,
                "run_metadata": props["run_metadata"],
                "software": {
                    "torch": props["torch_version"],
                    "torch_cuda": props["torch_cuda"],
                    "transformers": props["transformers_version"],
                },
                "hardware": {
                    "gpu_name": gpu_name,
                    "gpu_total_memory_bytes": gpu_mem,
                },
                "seed": seed,
                "total_unique_parameters": loaded.total_parameters,
            }
            t_wall0 = time.perf_counter()
            try:
                image = Image.open(ex["image_relpath"]).convert("RGB")
                out = run_one(
                    loaded,
                    image=image,
                    question=ex["question"],
                    max_new_tokens=int(cfg.get("max_new_tokens", 64)),
                    instruction=str(cfg.get("instruction")),
                )
                record.update(out)
                record["reference_answers_local"] = answers_by_qid.get(qid)
                n_ok += 1
                results_preview.append(
                    {
                        "question_id": qid,
                        "question": ex["question"],
                        "reference_answers": answers_by_qid.get(qid),
                        "prediction": out["prediction"],
                    }
                )
                print(
                    f"ok qid={qid} sec={out['generation_seconds']:.3f} "
                    f"pred={out['prediction']!r}"
                )
            except Exception as exc:  # noqa: BLE001
                record["status"] = "error"
                record["error"] = repr(exc)
                n_err += 1
                print(f"ERROR qid={qid}: {exc}", file=sys.stderr)
            record["wall_seconds"] = time.perf_counter() - t_wall0
            fout.write(json.dumps(record) + "\n")
            fout.flush()

    summary = {
        **props,
        "n_examples": len(examples),
        "n_ok": n_ok,
        "n_err": n_err,
        "n_skipped_existing": len(done),
        "out_jsonl": str(out_path),
        "preview": results_preview,
        "note": "Smoke diagnostics only; not a fair benchmark.",
    }
    Path(args.summary_json).write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {args.summary_json}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
