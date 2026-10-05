# P2 — Interactive InternVL document QA demo

**Label:** supplementary tooling after original final evaluation / P1.  
Does **not** change deployment selection or official Phase 4 claims.

## Features

- CLI: `companion/pascal/p2/cli.py` — single image + question → InternVL response  
- Browser UI: Gradio (`companion/pascal/p2/app.py`) — upload/preview, question, Run, response, caps, tiles, request timing, runtime details  
- Caps: **12** default; **4** and **1** labeled reduced visual budgets  
- Frozen instruction; greedy; `max_new_tokens=64`; no prompt editor; no output-based retries  
- Model loaded once per process; GPU requests serialized with a lock  
- No P1 memory-reset monkey-patch; live UI omits memory peaks  
- Development presets (illustrative); references only in expandable area **after** generation  
- Optional offline P1 viewer clearly labeled **prerecorded**

## Environment

Reuse `$HOME/.conda/envs/vlm_doc_pascal_p1` (P1 ML stack). UI pins:

```bash
export PYTHONNOUSERSITE=1
"$HOME/.conda/envs/vlm_doc_pascal_p1/bin/pip" install -r companion/pascal/requirements-ui.txt
```

Do not upgrade torch/transformers for UI packages.

## Runtime

Retains executed P1 dtype: **bfloat16** + eager attention.  
On V100, label BF16 as tensor usability unless `is_bf16_supported(including_emulation=False)` is True / CC≥8. Not claimed performance-optimal. See `P1_RUNTIME_CLARIFICATION.md`.

**Request timing scope:** `image_open_through_preprocess_transfer_generate_decode_cuda_synchronized`.

## CLI

```bash
export PYTHONNOUSERSITE=1
cd /home/ali.rasekh/teth/vlm_doc   # or your clone path
# list presets
"$HOME/.conda/envs/vlm_doc_pascal_p1/bin/python" companion/pascal/p2/cli.py --list-presets
# live inference (requires GPU allocation)
"$HOME/.conda/envs/vlm_doc_pascal_p1/bin/python" companion/pascal/p2/cli.py \
  --preset exact_success --max-num 12 --json
# arbitrary image
"$HOME/.conda/envs/vlm_doc_pascal_p1/bin/python" companion/pascal/p2/cli.py \
  --image path/to/page.png --question "What is the title?" --max-num 12
```

## UI allocation (Slurm)

```bash
sbatch companion/pascal/scripts/p2_demo.sbatch
# watch logs for hostname / port (default 7860)
```

Bind: **127.0.0.1 only**; `share=False`; no public tunnels.

### SSH forwarding

The Gradio app binds **127.0.0.1 only** on the compute node (`share=False`). Discover `COMPUTE_HOSTNAME` from the job log. On-site username: `ali.rasekh`. When already on the cluster network the login host is typically `pascal.l3s.intra`; use your real entry hostname for `LOGIN_HOST` if different. Do **not** invent an external FQDN.

**Preferred (laptop): jump host + loopback forward on the compute endpoint**

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -J ali.rasekh@LOGIN_HOST \
  -L 127.0.0.1:7860:127.0.0.1:7860 \
  ali.rasekh@COMPUTE_HOSTNAME
# Then open http://127.0.0.1:7860/ on the laptop
```

The `-L` destination is evaluated on the **final** SSH hop (compute node), so `127.0.0.1:7860` is the demo bound there.

**Two-stage alternative** (distinct intermediate port on the login node):

```bash
# on laptop:
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:7860:127.0.0.1:17860 \
  ali.rasekh@LOGIN_HOST
# on login node:
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:17860:127.0.0.1:7860 \
  ali.rasekh@COMPUTE_HOSTNAME
```

P2.1 corrected these instructions and the `p2_demo.sbatch` banner; a full external laptop→cluster tunnel was not tested from this agent session.

## Presets (selection rules)

| id | Rule |
|----|------|
| `exact_success` | P1 gallery unchanged; control ANLS=1 (qid 292) |
| `failure` | Smallest qid with P1 max_num=12 ANLS=0, nonempty pred (qid 5169) |
| `reduced_budget_regression` | P1 gallery regression at cap 4 (qid 5170) |
| `tile_reduction` | P1 gallery largest tile drop (qid 15325) |

Illustrative only — not a representative accuracy sample.

## Interview walkthrough (≈3 minutes)

1. Allocate demo job; open UI via SSH forward.  
2. Load **Exact success** preset; Run at cap 12; note tiles + request time.  
3. Switch cap to **4**; Run again; note tile drop / possible answer change.  
4. Load **Failure** preset; Run; expand references after generation only.  
5. Optionally open offline accordion for prerecorded P1 output (labeled).

## Limits

- Development/demo tooling only; no holdout.  
- Upload limits: 12 MiB and 40M pixels (reject; no silent resize).  
- Arbitrary uploads: no invented references/scores.  
- Independent requests (no chat history).  
- App-created temps (`p2_upload_*` from PIL→bytes path) are unlinked after the request in `app.py`.
- Gradio `type="filepath"` uploads are managed by Gradio under its cache (typically `/tmp/gradio/`); the app does not delete those Gradio-owned copies after inference. Preset paths on disk are never deleted.
- References match by **image SHA256 + exact question** (path-agnostic for Gradio cache copies) and clear when image/question change.
