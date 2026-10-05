# Limitations (Phase 4 report)

- **Sampled DocVQA evaluation**, not the full official DocVQA benchmark or leaderboard protocol.
- **Native visual-token budgets differ** across SmolVLM and InternVL configured systems; gaps are not isolatable to architecture alone.
- **Possible pretrained exposure** of foundation models to DocVQA or related document data is not ruled out from public cards alone.
- **Single training seed** and a **bounded adaptation budget** (200 optimizer updates, LoRA on language_model q/k/v/o only).
- **Synthetic robustness subset** (54 questions / 16 UCSF groups) is a limited stress test, not a corruption benchmark.
- **Project-specific primary ANLS** (`anls_normalized_strict_v2`) uses project normalization; not claimed byte-identical to Pix2Struct or VLMEvalKit.
- **Historical v1 supervised double end-of-turn bug** produced superseded adaptation results; Phase 3C corrected labeling and repeated training once. This does **not** prove the bug caused all v1 development regression.
- Fine-tuning **did not** improve primary development ANLS selection (step 0 remains the deployment candidate). Holdout rankings must not be used to reselect checkpoints.
- Bootstrap CIs describe uncertainty under the UCSF-cluster resampling design for this fixed holdout, **not** all sources of uncertainty.
- Generation latency is **generate-only** (CUDA-synchronized), not end-to-end wall-clock.
