#!/usr/bin/env python3
"""Bounded InternVL generate-return verification (no reference answers).

Compares adapter decoding against the pinned official single-turn chat path
on two development examples. Documents returned token IDs and slicing rule.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image


def main() -> int:
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        print("ERROR: PYTHONNOUSERSITE=1 required", file=sys.stderr)
        return 2

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/models/internvl3_1b.yaml")
    parser.add_argument("--manifest", default="data/manifests/docvqa_dev_v1.json")
    parser.add_argument(
        "--question-ids",
        default="5158,292",
        help="Comma-separated development question IDs (no answers used)",
    )
    parser.add_argument(
        "--out",
        default="results/internvl_generate_return_probe.json",
    )
    parser.add_argument("--cache-dir", default=".hf_cache")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))

    import yaml

    from vlm_doc.adapters.internvl import (
        DEFAULT_INSTRUCTION,
        load_internvl,
        load_pixel_values,
        run_one_internvl,
        _build_query_and_ids,
    )

    cfg = yaml.safe_load(Path(args.config).read_text())
    qids = [int(x) for x in args.question_ids.split(",") if x.strip()]
    man = json.loads(Path(args.manifest).read_text())
    by_qid = {int(e["question_id"]): e for e in man["examples"]}
    for q in qids:
        if q not in by_qid:
            print(f"ERROR: question_id {q} not in manifest", file=sys.stderr)
            return 2

    cache = Path(args.cache_dir).resolve()
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HF_HUB_CACHE", str(cache / "hub"))

    loaded = load_internvl(
        model_id=cfg["model_id"],
        revision=cfg["revision"],
        prefer_bf16=bool(cfg.get("prefer_bf16", True)),
        use_flash_attn=bool(cfg.get("use_flash_attn", False)),
        max_num=int(cfg.get("max_num", 12)),
        use_thumbnail=bool(cfg.get("use_thumbnail", True)),
    )

    # Static transformers evidence (inputs_embeds → empty starter input_ids).
    from transformers.generation.utils import GenerationMixin

    init_src = GenerationMixin._maybe_initialize_input_ids_for_generation
    init_text = init_src.__doc__ or ""
    # Read source lines for the empty-tensor branch
    import inspect

    init_code = inspect.getsource(init_src)
    has_empty_embeds_branch = (
        'torch.ones((batch_size, 0)' in init_code
        and "inputs_embeds" in init_code
    )

    remote_generate = inspect.getsource(loaded.model.generate)
    uses_inputs_embeds = (
        "inputs_embeds=input_embeds" in remote_generate
        or "inputs_embeds=input_embeds," in remote_generate
    )
    passes_input_ids_to_lm = "input_ids=" in remote_generate.split(
        "self.language_model.generate"
    )[-1][:400]

    results = []
    for qid in qids:
        ex = by_qid[qid]
        image = Image.open(ex["image_relpath"]).convert("RGB")
        question = str(ex["question"])

        # --- Adapter path (same as production) ---
        adapter_out = run_one_internvl(
            loaded,
            image=image,
            question=question,
            max_new_tokens=int(cfg.get("max_new_tokens", 64)),
            instruction=str(cfg.get("instruction", DEFAULT_INSTRUCTION)),
        )

        # Recompute shared preprocessing for official chat comparison.
        pixel_values, tile_info = load_pixel_values(
            image,
            input_size=loaded.image_size,
            max_num=loaded.max_num,
            use_thumbnail=loaded.use_thumbnail,
        )
        pixel_values = pixel_values.to(device=loaded.device, dtype=loaded.dtype)
        num_patches = int(pixel_values.shape[0])

        instruction = str(cfg.get("instruction", DEFAULT_INSTRUCTION))
        text = f"{question.strip()}\n{instruction}"
        question_marked = f"<image>\n{text}"
        query, input_ids, attention_mask, eos_token_id = _build_query_and_ids(
            loaded,
            question_with_image_marker=question_marked,
            num_patches=num_patches,
        )
        input_len = int(input_ids.shape[-1])
        gen_kwargs = {
            "max_new_tokens": int(cfg.get("max_new_tokens", 64)),
            "do_sample": False,
            "eos_token_id": eos_token_id,
        }

        with torch.inference_mode():
            generation_output = loaded.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
        out_ids = generation_output[0].detach().cpu()
        out_list = [int(x) for x in out_ids.tolist()]
        out_len = len(out_list)
        input_list = [int(x) for x in input_ids[0].detach().cpu().tolist()]

        # Evidence checks (never use output length alone).
        prefix_equal = (
            out_len >= input_len
            and out_list[:input_len] == input_list
        )
        # Returned IDs cannot equal prompt IDs when generate started from empty
        # input_ids under inputs_embeds (transformers empty starter).
        starts_with_prompt = prefix_equal
        # Special-token / prompt-text presence in decoded full return
        decoded_full = loaded.tokenizer.decode(out_ids, skip_special_tokens=True)
        decoded_full_keep_special = loaded.tokenizer.decode(
            out_ids, skip_special_tokens=False
        )
        question_in_decoded = question.strip() in decoded_full if question.strip() else False
        # IMG context token id should not appear in returned sequences if generated-only
        img_ctx_id = loaded.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        n_img_ctx_in_return = sum(1 for t in out_list if t == img_ctx_id)

        if starts_with_prompt:
            slicing_rule = "slice_at_input_len"
            decoded_sliced = loaded.tokenizer.decode(
                out_ids[input_len:], skip_special_tokens=True
            )
            convention = "prompt_plus_generated"
        else:
            slicing_rule = "decode_all_returned_ids_no_slice"
            decoded_sliced = decoded_full
            convention = "generated_only"

        # Official chat path with identical pixel_values + generation settings.
        chat_question = question_marked  # already has <image>\n
        # chat() auto-prepends <image>\n if missing; we pass the same marked question
        # but chat replaces <image> internally — match adapter text content.
        # Official API expects question possibly with <image>\n prefix.
        gen_cfg = {
            "max_new_tokens": int(cfg.get("max_new_tokens", 64)),
            "do_sample": False,
        }
        with torch.inference_mode():
            official_response = loaded.model.chat(
                tokenizer=loaded.tokenizer,
                pixel_values=pixel_values,
                question=chat_question,
                generation_config=dict(gen_cfg),
                history=None,
                return_history=False,
                num_patches_list=[num_patches],
            )

        # Adapter postprocess (sep split) already applied in run_one_internvl
        adapter_pred = adapter_out["prediction"]
        # Apply same sep strip to our raw decode for fair compare
        from vlm_doc.adapters import internvl_conv

        template = internvl_conv.get_conv_template(
            loaded.template_name or loaded.model.template, model=loaded.model
        )
        sep = template.sep.strip()
        probe_pred = decoded_sliced
        if sep and sep in probe_pred:
            probe_pred = probe_pred.split(sep)[0]
        probe_pred = probe_pred.strip()

        results.append(
            {
                "question_id": qid,
                "image_relpath": ex["image_relpath"],
                "question": question,
                "references_used": False,
                "input_token_length": input_len,
                "returned_token_length": out_len,
                "returned_token_ids_prefix": out_list[:32],
                "returned_token_ids_full_if_short": out_list if out_len <= 64 else None,
                "input_ids_prefix": input_list[:16],
                "starts_with_prompt_token_ids": starts_with_prompt,
                "n_img_context_tokens_in_return": n_img_ctx_in_return,
                "question_text_in_decoded_full": question_in_decoded,
                "decoded_full_skip_special": decoded_full,
                "decoded_full_keep_special_prefix": decoded_full_keep_special[:200],
                "slicing_rule": slicing_rule,
                "decoded_after_slicing_rule": probe_pred,
                "adapter_prediction": adapter_pred,
                "adapter_generate_return_convention_field": adapter_out.get(
                    "generate_return_convention"
                ),
                "official_chat_response": official_response,
                "adapter_matches_official_chat": adapter_pred == official_response,
                "probe_decode_matches_official_chat": probe_pred == official_response,
                "tile_info": {
                    "num_patches": num_patches,
                    "visual_token_count": int(
                        num_patches * loaded.num_image_token_per_tile
                    ),
                },
                "generation_settings": gen_kwargs,
                "verified_convention": convention,
            }
        )

    all_match = all(r["adapter_matches_official_chat"] for r in results)
    conventions = {r["verified_convention"] for r in results}
    verified = (
        conventions == {"generated_only"}
        and all(not r["starts_with_prompt_token_ids"] for r in results)
        and all(r["n_img_context_tokens_in_return"] == 0 for r in results)
        and all_match
    )

    report = {
        "task": "internvl_generate_return_probe",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": os.environ.get("HOSTNAME") or os.uname().nodename,
        "model_id": cfg["model_id"],
        "revision": cfg["revision"],
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "static_evidence": {
            "internvl_generate_passes_inputs_embeds_to_language_model": uses_inputs_embeds,
            "internvl_generate_passes_input_ids_to_language_model": passes_input_ids_to_lm,
            "transformers_inputs_embeds_starts_with_empty_input_ids": has_empty_embeds_branch,
            "official_chat_decodes_full_generation_output_without_prompt_slice": True,
            "note": (
                "InternVLChatModel.generate builds inputs_embeds from input_ids+vit, "
                "then calls language_model.generate(inputs_embeds=...) without input_ids. "
                "transformers GenerationMixin initializes input_ids as shape (batch, 0) "
                "when only inputs_embeds are provided; the returned LongTensor therefore "
                "contains newly generated token IDs only. Official chat batch_decodes "
                "that tensor with no input_len slice."
            ),
        },
        "verified": verified,
        "baseline_action": (
            "retain_existing_baseline"
            if verified
            else "fix_decode_and_supersede_or_rerun"
        ),
        "examples": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": str(out_path), "verified": verified, "n": len(results)}))
    for r in results:
        print(
            f"qid={r['question_id']} convention={r['verified_convention']} "
            f"in_len={r['input_token_length']} out_len={r['returned_token_length']} "
            f"match_official={r['adapter_matches_official_chat']} "
            f"pred={r['adapter_prediction']!r}"
        )
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
