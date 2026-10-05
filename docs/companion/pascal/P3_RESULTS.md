# P3 results — three-seed fixed-budget LoRA (development only)

**Label:** Supplementary development analysis after the original final holdout evaluation.  
**Does not** change deployment selection (InternVL step 0) or official Phase 4 claims.  
**No holdout. No best-seed selection.**

## Question

How variable is the corrected fixed-budget adaptation outcome across three training seeds on Pascal?

## Seeds and protocol

| Role | Seed |
|------|-----:|
| Original corrected v2 **S** | **42** |
| S+1 | **43** |
| S+2 | **44** |

Frozen protocol revision: `9470055e3c0bfae06a24fad984a6640ddacd1ff0` (`docs/companion/pascal/P3_PROTOCOL.md`).  
Continue launcher / wall-time amendment: `54b92a47b5c03bb59da4b1d8ca4bdb69818ff33b` (no scientific setting change).

Target mapping frozen (RNG-independent):  
`companion/pascal/artifacts/p3/frozen_qid_target_map.json` · `mapping_sha256=644000c1017f40678ee2a46f33606fd07cbd48ce9b07e02f28b8629b9bdf4e6a` · n=2000.

## Four-system development table (208 requested)

Rescored with `vlm_doc.metrics.score_example` + `aggregate_scores` (requested-set denominator). Failures/missing count as 0; generation-cap hits remain in the requested-set means (not dropped). None failed or missing here.

| System | ANLS | EM | ΔANLS vs fresh base | improved / regressed / tied | ok / fail / miss | gen-cap |
|--------|-----:|---:|-------------------:|----------------------------:|-----------------:|--------:|
| Fresh base | 0.8276 | 0.7644 | — | — | 208 / 0 / 0 | 0 |
| Seed 42 @ u200 | 0.8193 | 0.7644 | **−0.0084** | 5 / 6 / 197 | 208 / 0 / 0 | 0 |
| Seed 43 @ u200 | 0.8178 | 0.7596 | **−0.0098** | 8 / 7 / 193 | 208 / 0 / 0 | 0 |
| Seed 44 @ u200 | 0.8317 | 0.7740 | **+0.0040** | 6 / 3 / 199 | 208 / 0 / 0 | 0 |

Duplicate QIDs in outputs: **none**. Extra QIDs: **none**.

### Three-seed adapted ANLS summary (Pascal only)

| Statistic | Value |
|-----------|------:|
| Values | 0.8193, 0.8178, 0.8317 |
| Arithmetic mean | **0.8229** |
| Sample SD (ddof=1) across three seeds | **0.0076** |
| Min / Max | 0.8178 / 0.8317 |

The **0.0076** figure is the **sample standard deviation** of the three seed ANLS values (ddof=1). It is **not** a confidence interval and does not support a population-significance claim. Compact prose may write “0.8229 ± 0.0076 (sample SD, n=3 seeds)” with the same meaning.

Historical Neumann v2 u200 is **excluded** from mean/SD. Three seeds do **not** establish broad statistical significance. **Do not select seed 44.**

**Interpretation:** Two seeds fall below the fresh base and one is slightly above → **no consistent improvement** under this fixed protocol.

## Training counts and loss windows

Each completed run: **200** optimizer updates · **1600** presentations · **1600** unique QIDs (**80% coverage** of the 2000-example training pool under shuffle+wrap for these seeds; not full-pool coverage). Interrupted seed-44 attempt (164 updates, job 117234) is **preserved and excluded**.

| Seed | Job / node / GPU | Adapter SHA256 (safetensors) | mean loss first 20 | mean loss last 20 |
|-----:|------------------|------------------------------|-------------------:|------------------:|
| 42 | 117234 / pascal-node10 / V100-SXM3-32GB | `fb35e2213a1a9f012cebcf5d4633138901718671b15750c31bc56a8c7f1b246e` | 0.2701 | 0.2327 |
| 43 | 117234 / pascal-node10 / V100-SXM3-32GB | `3b0d90463a3e0640e63e7588fbe66ffd972c83800c179d670b8adfaca2c42e2f` | 0.2776 | 0.2171 |
| 44 | 117252 / pascal-node09 / V100-PCIE-32GB | `9ffea211147bece915a22cd8136b6aee6d90938db3fe114aba79929dc2778847` | 0.2944 | 0.1764 |

Loss curves label: training statistics on **potentially different examples** across seeds (order/membership digests differ).  
Figures: `companion/pascal/results/figures/p3_loss_curves.png`, `p3_dev_anls.png` (numeric agreement with table/logs verified; PNGs opened).

## Runtime / provenance audit

| Item | Finding |
|------|---------|
| Model revision | `OpenGVLab/InternVL3-1B` @ `4415a3b810e636d11dfa86b0e9ba40bb00535aa8` (all) |
| Train dtype policy | `prefer_bf16=true`, `use_flash_attn=false` (configs) |
| Eval dtype recorded | `bfloat16`; `bf16_supported=True` (default API; native CC≥8 **not** claimed) |
| Software | torch 2.5.1+cu118, transformers 4.48.3, peft 0.14.0, python 3.11.16 |
| Gate (117233) | V100-SXM3-32GB; BF16 default True; `including_emulation=False` **False**; all checks ok |
| **Protocol deviation** | Seeds **42/43 trained on pascal-node10 (V100-SXM3)**; seed **44 train + all four evals on pascal-node09 (V100-PCIE)**. Same GPU *family* (V100 32GB) and software, **not** the same physical GPU. |
| Attention backend | flash_attn not requested; inspectable eager path as in prior Pascal work (not re-probed per eval summary) |
| PEFT metadata | `adapter_config.json` inherits stale `base_model_name_or_path=./pretrained/Qwen2.5-32B-Instruct`. Load path constructs pinned InternVL3-1B and attaches LoRA via `load_language_lora` (project loader). |

### Timeout and seed-44 restart

- Job **117234** TIMEOUT after 04:00:22: seeds 42/43 complete; seed 44 stopped at **164/200** (no u200).  
- Interrupted tree preserved under `companion/pascal/artifacts/p3/interrupted/seed_44_job117234_*`.  
- Job **117252** restarted seed **44 from scratch** (fresh base load, fresh adapters/optimizer/RNG; identical seed=44 settings). Order digest matches the interrupted attempt (deterministic shuffle). No optimizer-state resume claimed.

## Fresh base vs P1 cap-12 control

| | ANLS | EM |
|--|-----:|---:|
| P1 cap-12 (job 117225) | 0.827624 | 0.764423 |
| P3 fresh base (job 117252) | 0.827624 | 0.764423 |
| Δ (fresh − P1) | **0** | **0** |

Exact agreement on primary means for this development set (diagnostic; not a claim of bit-identical predictions).

## Artifacts

| Path | Role |
|------|------|
| `companion/pascal/results/p3_seed_table.json` | Compact machine-readable table |
| `companion/pascal/results/p3_verified_analysis.json` | Rescore verification |
| `companion/pascal/results/p3_loss_curves.json` | Per-seed loss/LR series |
| `companion/pascal/results/figures/p3_*.png` | Loss + ANLS figures |
| `companion/pascal/artifacts/p3/seed_{42,43,44}/checkpoints/update_0200/` | Adapters (gitignored) |
| `companion/pascal/artifacts/p3/gate_ok.json` | Gate marker |

## Reproduction (outline)

```bash
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/companion/pascal:${PYTHONPATH:-}"
# after gate_ok exists:
sbatch companion/pascal/scripts/p3_main.sbatch
# or resume:
sbatch companion/pascal/scripts/p3_main_continue.sbatch
```

## Limitations

- Three seeds only; descriptive mean/SD; no population claim.  
- Seed variability includes shuffle membership/order, not init alone.  
- Cross-node / SXM3 vs PCIE V100 deviation for seed 44 vs 42/43.  
- BF16 on V100 is usability/emulation, not native CC≥8 acceleration.  
- No holdout evaluation of these adapters.  
- PEFT `base_model_name_or_path` metadata is stale; do not auto-load that path.
