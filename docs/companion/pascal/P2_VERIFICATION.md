# P2 / P2.1 verification record

## CPU (pre-GPU)

| Check | Result |
|-------|--------|
| `pytest companion/pascal/tests` | 13 passed (P2.1) |
| Import CLI/UI without CUDA | OK |
| Gradio `get_api_info` | **Restored** via `gradio_schema_patch.py` (bool `additionalProperties`); named endpoints include `/run_live`, `/apply_preset` |

## P2 GPU job 117228 (historical; preserved)

| Field | Value |
|-------|-------|
| State / elapsed | COMPLETED / 00:00:56 |
| Node / GPU | pascal-node10.l3s.intra / Tesla V100-SXM3-32GB |
| CLI vs engine | Prediction text matched, but stdout included `FlashAttention2 is not installed.` before JSON; `verify_job` used `rfind("{")` |
| UI | Loopback HTTP only; `get_api_info` stubbed empty |
| Browser visual QA | **Not** performed |
| Artifacts | `companion/pascal/artifacts/p2_verify/` (preserved); summary `results/p2_verification_summary.json` |

## P2.1 GPU job 117232

| Field | Value |
|-------|-------|
| Label | **P2.1** defect close-out |
| State / elapsed | **COMPLETED** / 00:01:30 |
| Node / GPU | pascal-node10.l3s.intra / Tesla V100-SXM3-32GB |
| Source at job start | `101d1b7…` + dirty P2.1 working tree (committed after) |
| dtype / attention | bfloat16 / eager; BF16 native (`including_emulation=False`) **False**; CC 7.0 |
| Strict CLI `json.loads` | **ok** — stdout pure JSON; FlashAttention notice on **stderr** only |
| CLI vs engine cap12 | **match** — `USMM 1/95-6/95, 12-Month Data`; tiles=13 |
| Cap isolation | engine + UI: cap12 tiles=13 → cap4 tiles=5 |
| Reference association | SHA256 + question; Gradio cache path `/tmp/gradio/…/qid_292.png` still matched `exact_success` |
| UI HTTP | ok `http://127.0.0.1:7865/` |
| Browser visual QA | **Passed** (Playwright headless Chromium): open → load preset → Run cap12 → Run cap4 → upload+question → empty-question error; 6 screenshots |
| Gradio Client API | ok (same actions; requires client-side schema patch) |
| SSH tunnel | Instructions corrected; external laptop→cluster tunnel **not** tested |
| Server / allocation | UI process terminated in-job; Slurm job ended; queue clear |

Compact summary: `companion/pascal/results/p2_1_verification_summary.json`  
Screenshots: `companion/pascal/results/p2_1_screenshots/`  
Raw: `companion/pascal/artifacts/p2_1_verify/` (gitignored).

### Gradio schema workaround (retained)

- Versions: `gradio==5.9.1`, `gradio_client==1.5.2`
- Cause: event schemas with `additionalProperties: true` (JSON boolean) break `get_type` / `_json_schema_to_python_type`
- Fix: patch those helpers to map bare bool schemas → `"Any"`; do **not** blank `get_api_info`
- ML stack not upgraded

### Upload / temp retention

- App-created `p2_upload_*` temps deleted after the request
- Gradio filepath cache copies under `/tmp/gradio/` retained by Gradio (observed in live JSON `image_meta.path`); not deleted by the app
- Preset files on disk never deleted
