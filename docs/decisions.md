# Decisions

| ID | Date | Decision | Rationale |
|----|------|----------|-----------|
| D1 | 2026-09-30 | Neumann-only; no Pascal dependency | User updated project decision; simplify ops |
| D2 | 2026-09-30 | PyTorch only; no TensorFlow | User decision; matches probe evidence (torch 2.5.1+cu118) |
| D3 | 2026-09-30 | Use ampere/`ampere`/`normal` only in Phase 1 | Prior association + job 81806; hopper/blackwell rejected; standby preemptible avoided |
| D4 | 2026-09-30 | Isolated env `vlm_doc` (Python 3.11); do not modify `gpu`/`qwen` | Reproducibility; avoid breaking other projects |
| D5 | 2026-09-30 | Pin torch 2.5.1+cu118 + transformers 4.48.3 | Conservative stack aligned with known-working CUDA 11.8 evidence and SmolVLM/Idefics3 support |
| D6 | 2026-09-30 | Standard attention; no FlashAttention requirement | Reduce compile/install risk for smoke |
| D7 | 2026-09-30 | Prefer Hub client download into project `.hf_cache` | Persistent authoritative cache; `/local` ephemeral only |
| D8 | 2026-09-30 | DocVQA smoke from `HuggingFaceM4/DocumentVQA` val shard | VLMEvalKit OpenCompass TSV HTTPS certificate expired; HF mirror is official DocVQA validation with stable IDs |
| D9 | 2026-09-30 | Evenly spaced `questionId`-sorted selection of 8 examples | Model-independent, reproducible; exclude from future holdout |
| D10 | 2026-09-30 | Force `PYTHONNOUSERSITE=1` | User-site packages polluted the new env (wrong torch/CUDA libs) |
| D11 | 2026-09-30 | Dedicated Ed25519 key + repo-local `core.sshCommand` | Push without changing global SSH or unrelated repos |
| D12 | 2026-09-30 | Full-validation frozen splits; group by `doc_id`; seed 42; ~200 dev / ~500 holdout | **Superseded by D20** for holdout grouping; development membership retained |
| D13 | 2026-09-30 | Primary ANLS = community lower/strip/collapse + NL threshold 0.5; keep VLMEvalKit variant secondary | **Superseded by D21** for primary cutoff; inclusive retained as legacy |
| D14 | 2026-09-30 | Resume requires matching run fingerprint | Prevent silent reuse across model/manifest/prompt/gen changes |
| D15 | 2026-09-30 | Report requested-set means (failures/missing → 0) | Failures must not inflate scores by exclusion |
| D16 | 2026-09-30 | No holdout inference / no training in Phase 2A | Development baseline only |
| D17 | 2026-09-30 | Primary ANLS uses `sim >= 0.5` (keep NL == 0.5 as 0.5); document ≠ pix2struct `NL < 0.5` | Historical Phase 2A clarification; superseded as primary by D21 |
| D18 | 2026-09-30 | Publish generation-only latency scope explicitly | Avoid misreading as end-to-end |
| D19 | 2026-09-30 | Maintain a small real I/O examples gallery from smoke/dev only | Traceability for review; holdout sealed |
| D20 | 2026-09-30 | Group by UCSF source-document ID; holdout_v2 after 6 UCSF overlaps; page-level fallback for missing UCSF | `doc_id` spans incomplete PDFs (178 multi-doc UCSFs); keep fixed 208 development IDs |
| D21 | 2026-09-30 | Primary metric `anls_normalized_strict_v2` (NL &lt; 0.5); CPU rescore; inclusive_v1 legacy | Align primary cutoff with verified pix2struct strict gate @ `19e41996…`; not claimed identical normalization |
| D22 | 2026-09-30 | Inference fingerprint `inference_v2` separate from scoring provenance | Docs-only commits must not invalidate inference; metric-only changes allow CPU rescore |
| D23 | 2026-09-30 | `PYTHONNOUSERSITE=1` only effective when set before interpreter start | Remove misleading in-process claim |
| D24 | 2026-09-30 | No holdout inference / no training / no score-driven prompt tuning in Phase 2B | Corrections + 500M development baseline only |
| D25 | 2026-09-30 | Terminal-period EM is diagnostic only; describe rank reversal as formatting sensitivity | Not official EM; primary scores retained separately |
| D26 | 2026-09-30 | Add InternVL3-1B as cross-family third development comparator; pin immutable revision; measure total unique params &lt;1e9 | Not a claim of newest/best sub-1B; official 448/`max_num=12`/thumbnail recipe; FlashAttn off |
| D27 | 2026-09-30 | Install `timm`+`einops` into existing `vlm_doc` after evidenced InternVL ImportError | Avoid separate env unless broader conflicts; do not upgrade torch/transformers |
| D28 | 2026-10-02 | InternVL generate return = `generated_only` (verified); retain Phase 2C baseline | Probe job 82527 + transformers empty `input_ids` under `inputs_embeds`; official chat has no prompt slice |
| D29 | 2026-10-02 | Train subset `docvqa_train_subset_v1` (~2000, seed 42) from pinned DocVQA train; exclude smoke/dev/holdout_v2 UCSF/page/hashes; target = first nonempty source answer | Leakage-checked prep only; no training loop; PAGE_FALLBACK never collapses missing IDs |
| D30 | 2026-10-02 | Phase 2B/2C review packets at `203b51f` were identical snapshots differing by header label only; regenerate phase02c packet after Phase 2D | Document intentional duplicate; do not treat byte-equal packets as independent inventories |
| D31 | 2026-10-02 | Phase 3A: LoRA only on InternVL `language_model` q/k/v/o_proj; freeze vision+mlp; r=16/α=32/drop=0.05; 200 AdamW updates | Language-side adaptation PoC; not a claim of better visual resolution |
| D32 | 2026-10-02 | Select LoRA checkpoint by development ANLS among steps 0/100/200; prefer earlier on ties; name best trained among 100/200 | Predeclared; if step 0 wins, report FT did not improve selection |
| D33 | 2026-10-02 | Ampere accepts CPU-only jobs (probe 82978); run train-image audit without GPU | Avoid idle-GPU data prep; GPU job starts after audit artifact exists |
| D34 | 2026-10-02 | Freeze deployment candidate = InternVL3-1B step 0; best trained = LoRA u100; u200 retained only | Phase 3A selection; no further FT/prompt/metric search in 3B |
| D35 | 2026-10-02 | Phase 3B robustness: 16 UCSF groups × ≤4 Q (seed 42); clean / half_detail BICUBIC / jpeg_q40 | Limited synthetic test; not a corruption benchmark; holdout sealed |
| D36 | 2026-10-02 | Prepare final holdout protocol for 256M/500M/InternVL/LoRA u100; UCSF-cluster bootstrap 2000× seed 42 | Do not execute until train-source review + explicit authorization |
| D37 | 2026-10-02 | Phase 3C: fix supervised double end-of-turn; repeat LoRA once as v2 with identical recipe | Implementation correction, not HPO; v1 preserved as historical |
| D38 | 2026-10-02 | Final-protocol adaptation comparison = LoRA v2 update 200 (best corrected trained); deploy candidate remains step 0 | Phase 3C selection; holdout still sealed |
| D39 | 2026-10-03 | Execute Phase 4 holdout under frozen protocol at evaluation revision `a9b0695…`; do not reselect checkpoints from holdout; package compact evidence only | Authorized final evaluation; CI includes zero for InternVL Δ ANLS; jsonl/weights stay local |
