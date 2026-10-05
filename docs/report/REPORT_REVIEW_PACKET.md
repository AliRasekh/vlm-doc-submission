# Report review packet (Phase 5 editorial pass)

## Paths

- PDF: `reports/vlm_doc_technical_report.pdf`
- Source: `reports/vlm_doc_technical_report.tex`, `reports/references.bib`
- Build: `reports/README.md`
- Submission: `docs/report/SUBMISSION_GUIDE.md`

## Assessment scope

**Not checked against the original assessment text** (assessment file / `AGENTS.md` absent from the repository). Absence of `AGENTS.md` is not treated as a delivery blocker; no artificial guidance file was created.

## Editorial corrections in this pass

1. Architecture naming from official InternVL3-1B model card: vision `InternViT-300M-448px-V2_5`, language `Qwen2.5-0.5B`; `Qwen2ForCausalLM` labeled as Transformers class only; card cited.
2. Evidence→hypothesis paragraph from development observations; model-selection eligibility paragraph; observations/hypotheses/outcomes separated.
3. Self-contained ANLS/EM definition; compact data-role table (manifest-verified); accurate split-construction history.
4. Training details (AdamW, linear warmup schedule, wd, clip, bf16, target rule, assistant-only masks); half-detail/JPEG definitions; loss means rounded in prose.
5. Presentation: loss ticks; robustness line/dot with truncated-axis note; evidence crops + marked full pages; latency/memory appendix table.
6. Exact CLI in SUBMISSION_GUIDE; local adapter ZIP under `artifacts/` (gitignored) + tracked SHA256 sidecar.

## Figure inventory

| Figure | Path |
|--------|------|
| Architecture | `docs/report/figures/architecture_lora_placement.svg` |
| Pipeline | `docs/report/figures/pipeline_flow.svg` |
| Train loss v2 | `docs/report/figures/train_loss_v2.svg` |
| Robustness v2 | `docs/report/figures/robustness_v2_anls.svg` |
| Holdout ANLS | `docs/report/figures/holdout_primary_anls.svg` |
| Paired CI | `docs/report/figures/paired_adaptation_ci.svg` |
| Evidence crops | `docs/report/gallery_examples/crops/` |

## Adapter package

- Path: `artifacts/internvl3_1b_lora_v2_u200_adapter.zip` (local/gitignored)
- SHA256 sidecar: `artifacts/internvl3_1b_lora_v2_u200_adapter_sha256.txt`
- Contents: adapter weights/config, checkpoint meta, pinned base ID/revision, checksums, loading README
- Excludes: base weights, datasets, credentials, optimizer state

## Validation

| Check | Outcome |
|-------|---------|
| Numeric cross-check | **PASS** (`results/phase05_report_numeric_check.json`) |
| Visual page inspection (10 pages) | **PASS** (`results/phase05_visual_inspection.json`) |
| Exact-command / path verification | **PASS** (`results/phase05_exact_command_verification.json`) |
| Assessment text | **Not checked against the original assessment text** |

## Adapter ZIP

- `artifacts/internvl3_1b_lora_v2_u200_adapter.zip` (local/gitignored)
- SHA256 sidecar: `artifacts/internvl3_1b_lora_v2_u200_adapter_sha256.txt` (updated when ZIP docs change; weights SHA256 remains `3c2c34cd…c344180a`)
- Metadata audit: `docs/report/adapter_metadata_audit.md`, `results/adapter_base_model_name_audit.json`
- `adapter_config.json` field `base_model_name_or_path=./pretrained/Qwen2.5-32B-Instruct` is inherited Hub `llm_config` metadata; supported load constructs InternVL3-1B then attaches to `language_model` only.
