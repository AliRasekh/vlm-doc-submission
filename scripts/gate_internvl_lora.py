#!/usr/bin/env python3
"""Phase 3A correctness gate before main InternVL LoRA training.

Checks: label masking, multimodal token counts, finite loss, nonzero LoRA grads,
frozen base unchanged, tiny memorization diagnostic, save/reload logits+greedy.
Discards diagnostic adapters afterward (caller reloads fresh base for main run).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import yaml
from PIL import Image


def main() -> int:
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        print("ERROR: PYTHONNOUSERSITE=1 required", file=sys.stderr)
        return 2

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/internvl3_1b_lora_v1.yaml")
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HOME", ".hf_cache"))
    parser.add_argument("--out", default="results/internvl_lora_gate_ok.json")
    parser.add_argument("--mem-updates", type=int, default=30)
    parser.add_argument("--mem-n", type=int, default=8)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))
    cache = Path(args.cache_dir).resolve()
    os.environ["HF_HOME"] = str(cache)
    os.environ.setdefault("HF_HUB_CACHE", str(cache / "hub"))

    cfg = yaml.safe_load(Path(args.config).read_text())
    if not torch.cuda.is_available():
        print("ERROR: CUDA required", file=sys.stderr)
        return 2

    from vlm_doc.adapters.internvl import load_internvl, run_one_internvl
    from vlm_doc.train import (
        attach_language_lora,
        build_supervised_example,
        label_mask_excerpt,
        load_language_lora,
        save_language_lora,
    )
    from vlm_doc.train.collator import assert_supervised_termination
    from vlm_doc.adapters import internvl_conv

    report: dict = {
        "task": "internvl_lora_correctness_gate",
        "phase": "3C_termination_fix" if "v2" in str(args.config) else "3A_legacy_or_other",
        "checks": {},
    }

    loaded = load_internvl(
        cfg["model_id"],
        cfg["revision"],
        prefer_bf16=True,
        use_flash_attn=False,
        max_num=int(cfg.get("max_num", 12)),
        use_thumbnail=bool(cfg.get("use_thumbnail", True)),
    )
    inv = attach_language_lora(
        loaded.model,
        r=int(cfg["lora_r"]),
        alpha=int(cfg["lora_alpha"]),
        dropout=float(cfg["lora_dropout"]),
        bias=str(cfg.get("lora_bias", "none")),
    )
    report["lora_inventory"] = {
        "pre_wrap_language_linear_targets": inv["pre_wrap_language_linear_targets"],
        "n_matched_lora_modules": inv["n_matched_lora_modules"],
        "trainable_parameters": inv["trainable_parameters"],
        "total_unique_parameters_including_adapters": inv[
            "total_unique_parameters_including_adapters"
        ],
        "trainable_fraction": inv["trainable_fraction"],
        "lora_config": inv["lora_config"],
        "all_trainable_under_language_model_lora": True,
    }
    print("GATE_LORA_MODULES", len(inv["pre_wrap_language_linear_targets"]))
    for n in inv["pre_wrap_language_linear_targets"]:
        print("  TARGET", n)

    if bool(cfg.get("gradient_checkpointing", True)):
        if hasattr(loaded.model.language_model, "gradient_checkpointing_enable"):
            loaded.model.language_model.gradient_checkpointing_enable()
        if hasattr(loaded.model.language_model, "enable_input_require_grads"):
            loaded.model.language_model.enable_input_require_grads()
    loaded.model.config.use_cache = False

    man = json.loads(Path(cfg["train_manifest"]).read_text())
    examples = man["examples"]
    # Pick representative + longest-looking by sorting image file size
    sized = []
    for e in examples:
        p = Path(e["image_relpath"])
        sized.append((p.stat().st_size if p.is_file() else 0, e))
    sized.sort(key=lambda x: x[0])
    picks = [sized[len(sized) // 2][1], sized[-1][1]]  # median + largest file
    # Prefer known double-eos evidence qid when present in the train subset.
    for e in examples:
        if int(e["question_id"]) == 46350:
            picks.append(e)
            break
    # Deduplicate by qid preserving order
    seen_q = set()
    uniq_picks = []
    for e in picks:
        q = int(e["question_id"])
        if q not in seen_q:
            uniq_picks.append(e)
            seen_q.add(q)
    picks = uniq_picks

    # Resolve eos id from template (same as collator).
    tmpl = internvl_conv.get_conv_template(
        loaded.template_name or loaded.model.template, model=loaded.model
    )
    eos_str = tmpl.sep.strip()
    eos_id = int(loaded.tokenizer.convert_tokens_to_ids(eos_str))

    mask_ok = []
    for e in picks:
        img = Image.open(e["image_relpath"]).convert("RGB")
        target = str(e["target_answer"])
        se = build_supervised_example(
            loaded,
            image=img,
            question=e["question"],
            target_answer=target,
            instruction=str(cfg["instruction"]),
            max_seq_length=int(cfg["max_seq_length"]),
            question_id=int(e["question_id"]),
        )
        assert se.n_supervised_tokens > 0
        assert se.n_img_context == se.num_patches * loaded.num_image_token_per_tile
        assert_supervised_termination(
            se.labels,
            prompt_len=se.prompt_len,
            eos_id=eos_id,
            question_id=se.question_id,
        )
        # Explicit: last supervised token is eos; no trailing newline supervised.
        lab = se.labels.view(-1)
        sup = lab[lab != -100]
        assert int(sup[-1]) == eos_id
        nl_id = int(loaded.tokenizer.encode("\n", add_special_tokens=False)[-1])
        assert nl_id not in set(int(x) for x in sup.tolist()), (
            f"qid={se.question_id}: supervised labels contain newline token"
        )
        excerpt = label_mask_excerpt(se.labels, loaded.tokenizer)
        assert excerpt.count("<|im_end|>") == 1, excerpt
        assert "Ċ <|im_end|>" not in excerpt and "<|im_end|> Ċ <|im_end|>" not in excerpt
        mask_ok.append(
            {
                "question_id": se.question_id,
                "seq_len": se.seq_len,
                "n_supervised": se.n_supervised_tokens,
                "n_img_context": se.n_img_context,
                "num_patches": se.num_patches,
                "excerpt": excerpt,
                "terminal_eos_ok": True,
                "single_im_end_in_excerpt": True,
            }
        )
    report["checks"]["label_masking"] = {"ok": True, "examples": mask_ok}
    report["checks"]["termination_single_eos"] = {
        "ok": True,
        "eos_token": eos_str,
        "eos_id": eos_id,
        "note": (
            "Phase 3C: assistant terminator located in assistant suffix; "
            "exactly one supervised <|im_end|>; no supervised trailing newline."
        ),
    }

    # Snapshot a frozen base parameter
    frozen_snap = None
    frozen_name = None
    for n, p in loaded.model.named_parameters():
        if not p.requires_grad and p.ndim >= 1 and p.numel() > 1000:
            frozen_name = n
            frozen_snap = p.detach().float().cpu().clone()
            break

    # Forward/backward finite + nonzero adapter grads
    e0 = picks[0]
    se = build_supervised_example(
        loaded,
        image=Image.open(e0["image_relpath"]).convert("RGB"),
        question=e0["question"],
        target_answer=str(e0["target_answer"]),
        instruction=str(cfg["instruction"]),
        max_seq_length=int(cfg["max_seq_length"]),
        question_id=int(e0["question_id"]),
    )
    loaded.model.train()
    loaded.model.zero_grad(set_to_none=True)
    out = loaded.model(
        pixel_values=se.pixel_values.to(loaded.device, dtype=loaded.dtype),
        input_ids=se.input_ids.to(loaded.device),
        attention_mask=se.attention_mask.to(loaded.device),
        image_flags=se.image_flags.to(loaded.device).unsqueeze(-1),
        labels=se.labels.to(loaded.device),
        use_cache=False,
        return_dict=True,
    )
    assert out.loss is not None and torch.isfinite(out.loss)
    out.loss.backward()
    any_grad = False
    for n, p in loaded.model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None
            if torch.any(p.grad != 0):
                any_grad = True
        else:
            assert p.grad is None or not torch.any(p.grad != 0)
    assert any_grad
    if frozen_snap is not None:
        for n, p in loaded.model.named_parameters():
            if n == frozen_name:
                assert torch.allclose(p.detach().float().cpu(), frozen_snap)
    report["checks"]["forward_backward"] = {
        "ok": True,
        "loss": float(out.loss.detach().item()),
        "nonzero_adapter_grads": True,
        "frozen_param_unchanged": frozen_name,
    }

    # Memorization diagnostic on 8 fixed train examples
    rng = random.Random(42)
    mem_pool = list(examples)
    rng.shuffle(mem_pool)
    mem_ex = mem_pool[: int(args.mem_n)]
    optim = torch.optim.AdamW(
        [p for p in loaded.model.parameters() if p.requires_grad],
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    losses = []
    for step in range(int(args.mem_updates)):
        optim.zero_grad(set_to_none=True)
        step_losses = []
        for e in mem_ex:
            se = build_supervised_example(
                loaded,
                image=Image.open(e["image_relpath"]).convert("RGB"),
                question=e["question"],
                target_answer=str(e["target_answer"]),
                instruction=str(cfg["instruction"]),
                max_seq_length=int(cfg["max_seq_length"]),
                question_id=int(e["question_id"]),
            )
            o = loaded.model(
                pixel_values=se.pixel_values.to(loaded.device, dtype=loaded.dtype),
                input_ids=se.input_ids.to(loaded.device),
                attention_mask=se.attention_mask.to(loaded.device),
                image_flags=se.image_flags.to(loaded.device).unsqueeze(-1),
                labels=se.labels.to(loaded.device),
                use_cache=False,
                return_dict=True,
            )
            (o.loss / len(mem_ex)).backward()
            step_losses.append(float(o.loss.detach().item()))
        torch.nn.utils.clip_grad_norm_(
            [p for p in loaded.model.parameters() if p.requires_grad],
            float(cfg["grad_clip"]),
        )
        optim.step()
        losses.append(sum(step_losses) / len(step_losses))
        print(f"GATE_MEM step={step+1} loss={losses[-1]:.4f}", flush=True)

    gens = []
    loaded.model.eval()
    with torch.inference_mode():
        for e in mem_ex[:3]:
            r = run_one_internvl(
                loaded,
                image=Image.open(e["image_relpath"]).convert("RGB"),
                question=e["question"],
                max_new_tokens=64,
                instruction=str(cfg["instruction"]),
            )
            gens.append(
                {
                    "question_id": int(e["question_id"]),
                    "target": e["target_answer"],
                    "prediction": r["prediction"],
                }
            )
    report["checks"]["memorization"] = {
        "ok": losses[-1] < losses[0],
        "n_examples": len(mem_ex),
        "updates": int(args.mem_updates),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "generations": gens,
        "question_ids": [int(e["question_id"]) for e in mem_ex],
    }
    if not report["checks"]["memorization"]["ok"]:
        print("ERROR: memorization loss did not decrease", file=sys.stderr)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        return 1

    # Save/reload compare logits + greedy
    tmp = Path(tempfile.mkdtemp(prefix="vlm_doc_lora_gate_"))
    try:
        save_language_lora(loaded.model, str(tmp / "adapter"))
        # Fixed input logits before reload (eval)
        loaded.model.eval()
        se = build_supervised_example(
            loaded,
            image=Image.open(mem_ex[0]["image_relpath"]).convert("RGB"),
            question=mem_ex[0]["question"],
            target_answer=str(mem_ex[0]["target_answer"]),
            instruction=str(cfg["instruction"]),
            max_seq_length=int(cfg["max_seq_length"]),
            question_id=int(mem_ex[0]["question_id"]),
        )
        with torch.inference_mode():
            logits_a = loaded.model(
                pixel_values=se.pixel_values.to(loaded.device, dtype=loaded.dtype),
                input_ids=se.input_ids.to(loaded.device),
                attention_mask=se.attention_mask.to(loaded.device),
                image_flags=se.image_flags.to(loaded.device).unsqueeze(-1),
                use_cache=False,
                return_dict=True,
            ).logits.float().cpu()
            pred_a = run_one_internvl(
                loaded,
                image=Image.open(mem_ex[0]["image_relpath"]).convert("RGB"),
                question=mem_ex[0]["question"],
                instruction=str(cfg["instruction"]),
            )["prediction"]

        # Reload fresh base + adapter
        loaded2 = load_internvl(
            cfg["model_id"],
            cfg["revision"],
            prefer_bf16=True,
            use_flash_attn=False,
            max_num=int(cfg.get("max_num", 12)),
            use_thumbnail=bool(cfg.get("use_thumbnail", True)),
        )
        load_language_lora(loaded2.model, str(tmp / "adapter"))
        loaded2.model.eval()
        se2 = build_supervised_example(
            loaded2,
            image=Image.open(mem_ex[0]["image_relpath"]).convert("RGB"),
            question=mem_ex[0]["question"],
            target_answer=str(mem_ex[0]["target_answer"]),
            instruction=str(cfg["instruction"]),
            max_seq_length=int(cfg["max_seq_length"]),
            question_id=int(mem_ex[0]["question_id"]),
        )
        with torch.inference_mode():
            logits_b = loaded2.model(
                pixel_values=se2.pixel_values.to(loaded2.device, dtype=loaded2.dtype),
                input_ids=se2.input_ids.to(loaded2.device),
                attention_mask=se2.attention_mask.to(loaded2.device),
                image_flags=se2.image_flags.to(loaded2.device).unsqueeze(-1),
                use_cache=False,
                return_dict=True,
            ).logits.float().cpu()
            pred_b = run_one_internvl(
                loaded2,
                image=Image.open(mem_ex[0]["image_relpath"]).convert("RGB"),
                question=mem_ex[0]["question"],
                instruction=str(cfg["instruction"]),
            )["prediction"]

        # BF16 path: allow modest tolerance on logits
        max_abs = float((logits_a - logits_b).abs().max().item())
        logits_tol = 5e-2
        report["checks"]["save_reload"] = {
            "ok": max_abs <= logits_tol and pred_a == pred_b,
            "logits_max_abs_diff": max_abs,
            "logits_tolerance": logits_tol,
            "greedy_pred_before": pred_a,
            "greedy_pred_after": pred_b,
            "greedy_match": pred_a == pred_b,
        }
        if not report["checks"]["save_reload"]["ok"]:
            print("ERROR: save/reload mismatch", report["checks"]["save_reload"], file=sys.stderr)
            Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
            return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    report["ok"] = True
    report["note"] = (
        "Diagnostic optimizer/adapters discarded. Main training must reload "
        "original base with fresh adapters."
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print("GATE_OK", json.dumps({"ok": True, "mem_final": losses[-1], "logits_diff": max_abs}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
