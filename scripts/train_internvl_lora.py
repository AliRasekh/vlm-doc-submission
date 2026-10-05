#!/usr/bin/env python3
"""Bounded InternVL3-1B language_model LoRA training (Phase 3A)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_targets(answers_path: Path, target_field: str) -> dict[int, str]:
    obj = json.loads(answers_path.read_text())
    records = obj.get("answers") or obj.get("records") or obj
    out: dict[int, str] = {}
    if isinstance(records, list):
        for r in records:
            qid = int(r["question_id"])
            if target_field in r and str(r[target_field]).strip():
                out[qid] = str(r[target_field]).strip()
            else:
                for a in r.get("answers") or []:
                    if str(a).strip():
                        out[qid] = str(a).strip()
                        break
    return out


def enable_input_require_grads_for_checkpointing(model) -> None:
    """Ensure LoRA gets grads when base embeddings are frozen under checkpointing."""
    emb = model.language_model.get_input_embeddings()

    def _req(grad_out):
        return grad_out

    # Hook on embeddings output is unreliable; PEFT recommends:
    if hasattr(model.language_model, "enable_input_require_grads"):
        model.language_model.enable_input_require_grads()
    else:
        def _make_inputs_require_grad(module, input, output):  # noqa: A002
            output.requires_grad_(True)

        emb.register_forward_hook(_make_inputs_require_grad)


def main() -> int:
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        print("ERROR: PYTHONNOUSERSITE=1 required before Python starts", file=sys.stderr)
        return 2

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/internvl3_1b_lora_v1.yaml")
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HOME", ".hf_cache"))
    parser.add_argument(
        "--skip-gate-marker",
        default="results/internvl_lora_gate_ok.json",
        help="If present, gate already passed; optional check only.",
    )
    parser.add_argument(
        "--overwrite-existing-outputs",
        action="store_true",
        help=(
            "Allow replacing an existing training log_jsonl and related run outputs. "
            "Without this flag, an existing log path fails safely."
        ),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))

    cache = Path(args.cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)
    os.environ.setdefault("HF_HUB_CACHE", str(cache / "hub"))

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(int(cfg["seed"]))

    # Fail safely before model load / any output mutation.
    log_path = Path(cfg["log_jsonl"])
    summary_path = Path(cfg.get("summary_json", "")) if cfg.get("summary_json") else None
    ex_md = Path(
        cfg.get(
            "supervised_example_md",
            "docs/examples/train_lora_v1_supervised_example.md",
        )
    )
    ex_json = Path(
        cfg.get(
            "supervised_example_json",
            "results/train_lora_v1_supervised_example.json",
        )
    )
    ckpt_root = Path(cfg["checkpoint_root"])
    save_updates = [int(x) for x in cfg.get("save_updates", [100, 200])]
    conflicting: list[str] = []
    if log_path.exists():
        conflicting.append(str(log_path))
    if summary_path is not None and summary_path.exists():
        conflicting.append(str(summary_path))
    if ex_md.exists():
        conflicting.append(str(ex_md))
    if ex_json.exists():
        conflicting.append(str(ex_json))
    for u in save_updates:
        upd = ckpt_root / f"update_{u:04d}"
        if upd.exists() and any(upd.iterdir()):
            conflicting.append(str(upd))
    if "lora_v1" in str(log_path) and "v2" in str(ckpt_root):
        print(
            f"ERROR: refusing to write v1 log path {log_path} from a v2 run",
            file=sys.stderr,
        )
        return 2
    if conflicting and not args.overwrite_existing_outputs:
        print(
            "ERROR: refusing to overwrite existing training outputs:\n  - "
            + "\n  - ".join(conflicting)
            + "\nPass --overwrite-existing-outputs for a documented overwrite, "
            "or point log/summary/example/checkpoint paths at a fresh run directory.",
            file=sys.stderr,
        )
        return 2

    if not torch.cuda.is_available():
        print("ERROR: CUDA required for training", file=sys.stderr)
        return 2

    from vlm_doc.adapters.internvl import load_internvl
    from vlm_doc.train import (
        attach_language_lora,
        build_supervised_example,
        label_mask_excerpt,
        save_language_lora,
    )

    loaded = load_internvl(
        cfg["model_id"],
        cfg["revision"],
        prefer_bf16=bool(cfg.get("prefer_bf16", True)),
        use_flash_attn=bool(cfg.get("use_flash_attn", False)),
        max_num=int(cfg.get("max_num", 12)),
        use_thumbnail=bool(cfg.get("use_thumbnail", True)),
    )
    inventory = attach_language_lora(
        loaded.model,
        r=int(cfg["lora_r"]),
        alpha=int(cfg["lora_alpha"]),
        dropout=float(cfg["lora_dropout"]),
        bias=str(cfg.get("lora_bias", "none")),
    )
    print("LORA_INVENTORY " + json.dumps({
        k: inventory[k]
        for k in (
            "pre_wrap_language_linear_targets",
            "n_matched_lora_modules",
            "total_unique_parameters_including_adapters",
            "trainable_parameters",
            "trainable_fraction",
            "lora_config",
        )
    }))

    if bool(cfg.get("gradient_checkpointing", True)):
        # Checkpoint LM; keep use_cache False.
        if hasattr(loaded.model.language_model, "gradient_checkpointing_enable"):
            loaded.model.language_model.gradient_checkpointing_enable()
        enable_input_require_grads_for_checkpointing(loaded.model)
    loaded.model.config.use_cache = False
    if hasattr(loaded.model.language_model, "config"):
        loaded.model.language_model.config.use_cache = False

    man = json.loads(Path(cfg["train_manifest"]).read_text())
    targets = load_targets(Path(cfg["train_answers"]), cfg.get("target_field", "target_answer"))
    examples = list(man["examples"])
    # Prefer manifest target_answer when present.
    for e in examples:
        qid = int(e["question_id"])
        if e.get("target_answer"):
            targets[qid] = str(e["target_answer"]).strip()
        elif qid not in targets:
            raise KeyError(f"missing target for qid={qid}")

    order = list(range(len(examples)))
    rng = random.Random(int(cfg["seed"]))
    rng.shuffle(order)

    # Document one real training example before the loop.
    demo_e = examples[order[0]]
    demo_img = Image.open(demo_e["image_relpath"]).convert("RGB")
    demo = build_supervised_example(
        loaded,
        image=demo_img,
        question=demo_e["question"],
        target_answer=targets[int(demo_e["question_id"])],
        instruction=str(cfg["instruction"]),
        max_seq_length=int(cfg["max_seq_length"]),
        question_id=int(demo_e["question_id"]),
    )
    demo_doc = {
        "question_id": demo.question_id,
        "question": demo.question,
        "target_answer": demo.target_answer,
        "num_patches": demo.num_patches,
        "seq_len": demo.seq_len,
        "prompt_len": demo.prompt_len,
        "n_supervised_tokens": demo.n_supervised_tokens,
        "n_img_context": demo.n_img_context,
        "label_mask_excerpt": label_mask_excerpt(
            demo.labels, loaded.tokenizer, max_show=48
        ),
        "supervised_token_ids": [
            int(x) for x in demo.labels.view(-1).tolist() if int(x) != -100
        ],
        "note": (
            "IMG_CONTEXT tokens summarized via n_img_context; excerpt compresses "
            "MASK runs rather than dumping thousands of image tokens."
        ),
    }
    ex_md = Path(
        cfg.get(
            "supervised_example_md",
            "docs/examples/train_lora_v1_supervised_example.md",
        )
    )
    ex_json = Path(
        cfg.get(
            "supervised_example_json",
            "results/train_lora_v1_supervised_example.json",
        )
    )
    ex_md.parent.mkdir(parents=True, exist_ok=True)
    ex_json.parent.mkdir(parents=True, exist_ok=True)
    phase_label = cfg.get("phase_label", "LoRA supervised example")
    ex_md.write_text(
        f"# Supervised training example ({phase_label})\n\n"
        f"**qid:** {demo_doc['question_id']}\n\n"
        f"**Question:** {demo_doc['question']}\n\n"
        f"**Target:** `{demo_doc['target_answer']}`\n\n"
        f"**Tiles (num_patches):** {demo_doc['num_patches']}  \n"
        f"**Seq len:** {demo_doc['seq_len']} (prompt_len={demo_doc['prompt_len']})  \n"
        f"**Supervised tokens:** {demo_doc['n_supervised_tokens']}  \n"
        f"**IMG_CONTEXT count:** {demo_doc['n_img_context']}\n\n"
        f"**Label-mask excerpt:**\n\n```\n{demo_doc['label_mask_excerpt']}\n```\n"
    )
    ex_json.write_text(json.dumps(demo_doc, indent=2) + "\n")

    trainable_params = [p for p in loaded.model.parameters() if p.requires_grad]
    optim = AdamW(
        trainable_params,
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    max_updates = int(cfg["max_updates"])
    warmup = int(cfg["warmup_updates"])
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=warmup, num_training_steps=max_updates
    )
    grad_accum = int(cfg["grad_accum"])
    grad_clip = float(cfg["grad_clip"])
    save_updates = set(int(x) for x in cfg.get("save_updates", [100, 200]))
    ckpt_root = Path(cfg["checkpoint_root"])
    ckpt_root.mkdir(parents=True, exist_ok=True)

    log_path = Path(cfg["log_jsonl"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite_existing_outputs and log_path.exists():
        log_path.unlink()

    loaded.model.train()
    optim.zero_grad(set_to_none=True)

    update = 0
    micro = 0
    cursor = 0
    samples_seen = 0
    unique_qids: set[int] = set()
    loss_window: list[float] = []
    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    initial_loss = None
    final_loss = None

    while update < max_updates:
        idx = order[cursor % len(order)]
        cursor += 1
        ex = examples[idx]
        qid = int(ex["question_id"])
        image = Image.open(ex["image_relpath"]).convert("RGB")
        batch = build_supervised_example(
            loaded,
            image=image,
            question=ex["question"],
            target_answer=targets[qid],
            instruction=str(cfg["instruction"]),
            max_seq_length=int(cfg["max_seq_length"]),
            question_id=qid,
        )
        samples_seen += 1
        unique_qids.add(qid)

        device = loaded.device
        dtype = loaded.dtype
        pixel_values = batch.pixel_values.to(device=device, dtype=dtype)
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        labels = batch.labels.to(device)
        image_flags = batch.image_flags.to(device)

        out = loaded.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=image_flags.unsqueeze(-1),
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        loss = out.loss
        if loss is None or not torch.isfinite(loss):
            print(f"ERROR: non-finite loss at update={update} qid={qid}", file=sys.stderr)
            return 2
        (loss / grad_accum).backward()
        loss_window.append(float(loss.detach().item()))
        micro += 1

        if micro % grad_accum == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
            # Verify adapter grads nonzero on first step
            if update == 0:
                any_grad = any(
                    (p.grad is not None and torch.any(p.grad != 0).item())
                    for p in trainable_params
                )
                if not any_grad:
                    print("ERROR: LoRA gradients are zero after first accum", file=sys.stderr)
                    return 2
                # Frozen base sample check
                for n, p in loaded.model.named_parameters():
                    if p.requires_grad:
                        continue
                    if p.grad is not None and torch.any(p.grad != 0):
                        print(f"ERROR: frozen param has grad: {n}", file=sys.stderr)
                        return 2

            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            update += 1
            mean_loss = sum(loss_window) / len(loss_window)
            loss_window = []
            if initial_loss is None:
                initial_loss = mean_loss
            final_loss = mean_loss
            lr_now = float(sched.get_last_lr()[0])
            peak = int(torch.cuda.max_memory_allocated())
            rec = {
                "update": update,
                "loss": mean_loss,
                "lr": lr_now,
                "grad_norm": float(grad_norm) if not isinstance(grad_norm, float) else float(grad_norm),
                "samples_seen": samples_seen,
                "unique_examples_seen": len(unique_qids),
                "elapsed_seconds": time.perf_counter() - t0,
                "peak_allocated_bytes": peak,
                "last_question_id": qid,
            }
            with log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(
                f"update={update}/{max_updates} loss={mean_loss:.4f} lr={lr_now:.2e} "
                f"unique={len(unique_qids)} peak_gib={peak/1024**3:.2f}",
                flush=True,
            )

            if update in save_updates:
                out_dir = ckpt_root / f"update_{update:04d}"
                save_language_lora(loaded.model, str(out_dir))
                # Weight hash of adapter
                weight_files = sorted(out_dir.glob("*.safetensors")) + sorted(
                    out_dir.glob("adapter_model.bin")
                )
                whashes = {p.name: sha256_file(p) for p in weight_files}
                meta = {
                    "update": update,
                    "adapter_weight_sha256": whashes,
                    "lora_inventory": inventory,
                    "train_config_sha256": sha256_file(Path(args.config)),
                    "train_manifest_examples_content_sha256": man.get(
                        "examples_content_sha256"
                    ),
                    "saved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
                (out_dir / "vlm_doc_checkpoint_meta.json").write_text(
                    json.dumps(meta, indent=2) + "\n"
                )
                print(f"SAVED_CHECKPOINT {out_dir}", flush=True)

    summary = {
        "task": (
            "train_internvl3_1b_lora_v2"
            if "lora_v2" in str(ckpt_root)
            else "train_internvl3_1b_lora_v1"
        ),
        "max_updates": max_updates,
        "completed_updates": update,
        "samples_seen": samples_seen,
        "unique_examples_seen": len(unique_qids),
        "presentations_note": (
            "samples_seen counts microbatch presentations; with shuffle+wrap this "
            "is not necessarily one full epoch over 2000 examples."
        ),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "elapsed_seconds": time.perf_counter() - t0,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "lora_inventory": inventory,
        "checkpoint_root": str(ckpt_root),
        "software": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "peft": __import__("peft").__version__,
        },
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": os.uname().nodename,
        "gpu": torch.cuda.get_device_name(0),
    }
    Path(cfg["summary_json"]).write_text(json.dumps(summary, indent=2) + "\n")
    print("TRAIN_DONE " + json.dumps({
        "updates": update,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "unique": len(unique_qids),
        "peak_gib": summary["peak_allocated_bytes"] / 1024**3,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
