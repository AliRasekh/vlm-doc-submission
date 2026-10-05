# Phase 5 — Technical report and submission package

**Status:** complete (report compiled from frozen Phase 4 evidence; no new training/inference).

## Packaging corrections

- Regenerated `architecture_lora_placement.svg` and `pipeline_flow.svg` as valid UTF-8 (ASCII labels).
- Main robustness figure: `docs/report/figures/robustness_v2_anls.svg` from base + LoRA v2 u200; v1 figure labeled historical.
- Loss-window documentation corrected to exact packaged means: first 20 = `0.25614097751677034`, last 20 = `0.21429072171449662`.
- Adapted parameter reporting: total unique including adapters `940,355,712`; trainable `2,162,688` (avoided ambiguous “938.2M+”).
- Language backbone naming: official card `Qwen2.5-0.5B`; Transformers class `Qwen2ForCausalLM` only as implementation detail; vision card name `InternViT-300M-448px-V2_5`.
- Editorial pass: self-contained ANLS definition, data-role table, evidence crops, exact submission commands, local adapter ZIP.
- Packaging correction: adapter `base_model_name_or_path` audited as inherited stale `llm_config` metadata (not 32B training); InternVL3 bibtex from official arXiv authors; delivery items clarified in SUBMISSION_GUIDE.

## Report

- Source: `reports/vlm_doc_technical_report.tex`
- PDF: `reports/vlm_doc_technical_report.pdf`
- Build: `reports/README.md`
- Submission pointers: `docs/report/SUBMISSION_GUIDE.md`
- Numeric check: `results/phase05_report_numeric_check.json`

## Visual verification

Pages rendered with `mutool draw` to `reports/page_previews/` (local; gitignored) and inspected for clipped text, broken glyphs, and figure readability.
