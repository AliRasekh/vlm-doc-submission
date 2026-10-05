# Sources

## Models

| Resource | Identifier | Revision / commit | Notes |
|----------|------------|-------------------|-------|
| SmolVLM-256M-Instruct | `HuggingFaceTB/SmolVLM-256M-Instruct` | `7e3e67edbbed1bf9888184d9df282b700a323964` | Public; measured 256,484,928 unique params; Idefics3 |
| SmolVLM-500M-Instruct | `HuggingFaceTB/SmolVLM-500M-Instruct` | `a7da5b986cb59b408707209984f360a5f4ad7e47` | Public; measured 507,482,304 unique params |
| InternVL3-1B | `OpenGVLab/InternVL3-1B` | `4415a3b810e636d11dfa86b0e9ba40bb00535aa8` | Cross-family comparator / deployment candidate (step 0); 938,193,024 unique params; Apache-2.0 |
| Model card / API | https://huggingface.co/OpenGVLab/InternVL3-1B | same SHA | Vision `InternViT-300M-448px-V2_5`; language `Qwen2.5-0.5B`; dynamic preprocess 448 / max_num=12 / thumbnail |
| InternVL LoRA v1 u100 (historical) | `checkpoints/internvl3_1b_lora_v1/update_0100` | adapter SHA256 `16564ec1…8f67c219` | Superseded for intended labels (double EOT bug); historical only |
| InternVL LoRA v1 u200 (historical) | `checkpoints/internvl3_1b_lora_v1/update_0200` | adapter SHA256 `4b925a85…bd36e1e7` | Historical; not selected |
| InternVL LoRA **v2 u200** (best corrected trained) | `checkpoints/internvl3_1b_lora_v2/update_0200` | adapter SHA256 `3c2c34cd…c344180a` | Phase 3C adaptation comparison; deploy candidate remains step 0 |

## Datasets

| Resource | Identifier | Revision / commit | Notes |
|----------|------------|-------------------|-------|
| DocVQA (HF mirror) | `HuggingFaceM4/DocumentVQA` | `a44195f8f989c10d06ab4ccc98ffe9cdb6d928e4` | Official DocVQA fields (`questionId`, `docId`, `ucsf_document_*`, `answers`, `image`); validation used for smoke/dev/holdout; **train** used for `docvqa_train_subset_v1` |
| VLMEvalKit DocVQA_VAL route | `open-compass/VLMEvalKit` | `54a063c5bf75245c35599daab9826d0a5b8e2671` | `vlmeval/dataset/image_vqa.py` lists `DocVQA_VAL` TSV URL + MD5 `d5ee77e1926ff10690d469c56b73eabf` |
| OpenCompass DocVQA_VAL TSV | `https://opencompass.openxlab.space/utils/VLMEval/DocVQA_VAL.tsv` | (not downloaded) | HTTPS failed: **certificate has expired**; TLS verification not disabled |

## Cluster evidence

| Item | Source |
|------|--------|
| Phase 0 audit | Local untracked `phase0_neumann.local.md` (2026-09-30 ~08:00+02 / re-check ~08:11+02) |
| Probe job 81806 | Slurm `sacct` / probe stdout (A100 80GB, ampere) |
| InternVL generate-return probe 82527 | `results/internvl_generate_return_probe.json` (A100-PCIE-40GB, gpunode05) |

## Software pins

See `environment/versions.verified.txt` and `environment/requirements.txt`.

## Metrics references

| Resource | Location | Notes |
|----------|----------|-------|
| VLMEvalKit `anls_compute` / DocVQA hit rule | `open-compass/VLMEvalKit` @ `54a063c5bf75245c35599daab9826d0a5b8e2671` `vlmeval/dataset/utils/vqa_eval.py` | NL on lowercased whitespace-normalized strings; denominator uses `max(len(gt.upper()), len(pred.upper()))` on originals; DocVQA keeps score iff similarity ≥ 0.5 |
| pix2struct strict cutoff (verified) | https://github.com/google-research/pix2struct/blob/19e419961b260791d1f3314faa32257ad1430bd9/pix2struct/metrics.py | Commit `19e419961b260791d1f3314faa32257ad1430bd9`: `score = 1 - nl if nl < 0.5 else 0` |
| LayoutLMv2 public ANLS | https://github.com/herobd/layoutlmv2 (eval helpers) | `1-NL if NL<0.5 else 0`; max over answers |
| MP-DocVQA-style evaluator | https://github.com/rubenpt91/MP-DocVQA-Framework/blob/master/metrics.py | lower+strip; threshold on similarity ≥ 0.5 |

**Primary (Phase 2B+):** `anls_normalized_strict_v2` — lowercase+strip+collapse; NL = edit / max(len_norm); score = 1−NL iff **NL &lt; 0.5** else 0; max over nonempty refs. Project metric; **not** claimed byte-identical to Pix2Struct (normalization differs). Strict cutoff cites pix2struct @ `19e41996…` above.

**Legacy:** `anls_normalized_inclusive_v1` (Phase 2A primary) keeps score when sim ≥ 0.5 (NL ≤ 0.5).  
**Compatibility:** `anls_vlmevalkit` (denominator quirk).
