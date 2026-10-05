#!/usr/bin/env python3
"""P3 main: three seeded trains + fresh-base/adapter dev evals + analysis."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from . import SEEDS
from .analyze import DEV_MANIFEST, N_DEV_EXPECTED, load_predictions_jsonl, load_requested_qids
from .analyze import analyze
from .eval_wrap import run_eval
from .paths import P3_ART, REPO_ROOT, seed_dirs
from .train_wrap import run_train

EXPECTED_MANIFEST_CONTENT_SHA256 = (
    "8dce2b86837c5ce52002acc802c8a45188d00bf1c45362bdf6112969a5e54744"
)
EXPECTED_CONFIG_SHA256 = (
    "0eb33646a2a3822b70ea5e993ebbf985bfc54a96ce809c7469f2681f70eacad9"
)
EXPECTED_MODEL_ID = "OpenGVLab/InternVL3-1B"
EXPECTED_MODEL_REVISION = "4415a3b810e636d11dfa86b0e9ba40bb00535aa8"
EXPECTED_MAX_NEW_TOKENS = 64
EXPECTED_INSTRUCTION = "Answer the question using a single word or short phrase."
EXPECTED_DO_SAMPLE = False
EXPECTED_PREFER_BF16 = True
EXPECTED_MAX_NUM = 12
EXPECTED_USE_THUMBNAIL = True


def _adapter_weight_sha(adapter_dir: Path) -> str | None:
    meta = adapter_dir / "vlm_doc_checkpoint_meta.json"
    if meta.is_file():
        m = json.loads(meta.read_text())
        wh = m.get("adapter_weight_sha256") or {}
        if "adapter_model.safetensors" in wh:
            return wh["adapter_model.safetensors"]
        if wh:
            return next(iter(wh.values()))
    return None


def _require(fp: dict[str, Any], key: str, problems: list[str]) -> Any:
    if key not in fp or fp.get(key) is None or fp.get(key) == "":
        problems.append(f"missing required fingerprint field: {key}")
        return None
    return fp.get(key)


def _adapter_sha_from_fp(fp: dict[str, Any]) -> str | None:
    got = fp.get("adapter_weights_sha256")
    if isinstance(got, dict):
        if not got:
            return None
        return got.get("adapter_model.safetensors") or next(iter(got.values()))
    if isinstance(got, str) and got:
        return got
    return None


def validate_eval_cache(
    *,
    tag: str,
    summary_path: Path,
    jsonl_path: Path,
    expected_adapter_dir: Path | None,
    expected_train_update: int | None,
) -> tuple[str, str]:
    """Return (status, message).

    status:
      - ok_complete: safe to skip
      - missing: no usable cache (caller may run)
      - incompatible: evidence present but invalid — preserve, do not overwrite/rerun
    """
    if not summary_path.is_file() and not jsonl_path.is_file():
        return "missing", f"{tag}: no summary or jsonl"
    if summary_path.is_file() and not jsonl_path.is_file():
        return (
            "incompatible",
            f"{tag}: summary exists without predictions jsonl — preserving; not re-running",
        )
    if jsonl_path.is_file() and not summary_path.is_file():
        return (
            "incompatible",
            f"{tag}: predictions jsonl exists without summary — preserving; not re-running",
        )

    try:
        summary = json.loads(summary_path.read_text())
    except Exception as exc:  # noqa: BLE001
        return "incompatible", f"{tag}: unreadable summary ({exc})"

    scores = summary.get("scores") or {}
    fp = summary.get("run_fingerprint")
    problems: list[str] = []

    if not isinstance(fp, dict) or not fp:
        problems.append("missing required run_fingerprint object")
        return "incompatible", f"{tag}: incompatible cached eval: " + "; ".join(problems)

    if int(scores.get("n_requested") or -1) != N_DEV_EXPECTED:
        problems.append(
            f"n_requested={scores.get('n_requested')} expected={N_DEV_EXPECTED}"
        )

    try:
        requested = load_requested_qids(DEV_MANIFEST)
        preds = load_predictions_jsonl(jsonl_path, requested_qids=requested)
    except Exception as exc:  # noqa: BLE001
        return "incompatible", f"{tag}: prediction integrity failure ({exc})"

    n_ok = sum(1 for q in requested if preds.get(q, {}).get("status") == "ok")
    n_fail = sum(
        1
        for q in requested
        if q in preds and preds[q].get("status", "ok") != "ok"
    )
    n_miss = sum(1 for q in requested if q not in preds)
    if n_ok + n_fail + n_miss != N_DEV_EXPECTED:
        problems.append("coverage arithmetic mismatch")
    if n_miss:
        problems.append(f"missing_qids={n_miss}")
    if "n_missing" not in scores:
        problems.append("summary scores missing n_missing")
    elif int(scores.get("n_missing") or 0) != n_miss:
        problems.append(
            f"summary n_missing={scores.get('n_missing')} != jsonl missing={n_miss}"
        )

    # Required provenance must be present (not only checked when nonempty).
    model_id = _require(fp, "model_id", problems)
    if model_id is not None and model_id != EXPECTED_MODEL_ID:
        problems.append(f"model_id={model_id}")
    if summary.get("model_id") != EXPECTED_MODEL_ID:
        problems.append(f"summary.model_id={summary.get('model_id')}")

    rev = _require(fp, "model_revision", problems)
    if rev is not None and rev != EXPECTED_MODEL_REVISION:
        problems.append(f"model_revision={rev}")
    if summary.get("model_revision") != EXPECTED_MODEL_REVISION:
        problems.append(f"summary.model_revision={summary.get('model_revision')}")

    man = _require(fp, "manifest_content_sha256", problems)
    if man is not None and man != EXPECTED_MANIFEST_CONTENT_SHA256:
        problems.append(f"manifest_content_sha256={man}")
    if summary.get("manifest_content_sha256") not in (
        None,
        EXPECTED_MANIFEST_CONTENT_SHA256,
    ):
        if summary.get("manifest_content_sha256") != EXPECTED_MANIFEST_CONTENT_SHA256:
            problems.append(
                f"summary.manifest_content_sha256={summary.get('manifest_content_sha256')}"
            )

    cfg = _require(fp, "config_sha256", problems)
    if cfg is not None and cfg != EXPECTED_CONFIG_SHA256:
        problems.append(f"config_sha256={cfg}")

    instr = _require(fp, "instruction", problems)
    if instr is not None and instr != EXPECTED_INSTRUCTION:
        problems.append("instruction mismatch")

    mnt = _require(fp, "max_new_tokens", problems)
    if mnt is not None and int(mnt) != EXPECTED_MAX_NEW_TOKENS:
        problems.append(f"max_new_tokens={mnt}")

    if "do_sample" not in fp:
        problems.append("missing required fingerprint field: do_sample")
    elif bool(fp.get("do_sample")) != EXPECTED_DO_SAMPLE:
        problems.append(f"do_sample={fp.get('do_sample')}")

    if "prefer_bf16" not in fp:
        problems.append("missing required fingerprint field: prefer_bf16")
    elif bool(fp.get("prefer_bf16")) != EXPECTED_PREFER_BF16:
        problems.append(f"prefer_bf16={fp.get('prefer_bf16')}")

    proc = _require(fp, "processor_settings", problems)
    if isinstance(proc, dict):
        if int(proc.get("max_num", -1)) != EXPECTED_MAX_NUM:
            problems.append(f"processor_settings.max_num={proc.get('max_num')}")
        if bool(proc.get("use_thumbnail")) != EXPECTED_USE_THUMBNAIL:
            problems.append(
                f"processor_settings.use_thumbnail={proc.get('use_thumbnail')}"
            )
    elif proc is not None:
        problems.append("processor_settings is not an object")

    fp_sha = _require(fp, "run_fingerprint_sha256", problems)

    # Prediction-record fingerprint coherence with summary
    sample_qids = [q for q in requested if q in preds][:5]
    if not sample_qids and n_ok + n_fail > 0:
        problems.append("no prediction rows available for fingerprint coherence")
    for qid in sample_qids:
        rec = preds[qid]
        rfp = rec.get("run_fingerprint")
        if not isinstance(rfp, dict) or not rfp:
            problems.append(f"qid={qid} missing run_fingerprint on prediction record")
            continue
        rec_sha = rfp.get("run_fingerprint_sha256")
        if not rec_sha:
            problems.append(f"qid={qid} missing run_fingerprint_sha256 on record")
        elif fp_sha is not None and rec_sha != fp_sha:
            problems.append(
                f"qid={qid} record fingerprint sha {rec_sha} != summary {fp_sha}"
            )

    # Adapter expectations
    if expected_adapter_dir is None:
        aw = fp.get("adapter_weights_sha256")
        if aw:
            problems.append("base eval unexpectedly records adapter weights")
        if fp.get("adapter_dir") not in (None, "", False):
            # tolerate absolute path only if weights empty; still flag nonzero dir with weights
            if aw:
                problems.append(f"base eval adapter_dir unexpected: {fp.get('adapter_dir')}")
        if expected_train_update is not None:
            problems.append("internal: base eval should not expect train_update")
        if fp.get("train_update") not in (None,):
            problems.append(f"base eval unexpected train_update={fp.get('train_update')}")
    else:
        if "adapter_weights_sha256" not in fp:
            problems.append("missing required fingerprint field: adapter_weights_sha256")
        if "train_update" not in fp or fp.get("train_update") is None:
            problems.append("missing required fingerprint field: train_update")
        if not expected_adapter_dir.is_dir():
            problems.append(f"expected adapter dir missing: {expected_adapter_dir}")
        else:
            exp_sha = _adapter_weight_sha(expected_adapter_dir)
            got_sha = _adapter_sha_from_fp(fp)
            if not got_sha:
                problems.append("adapter_weights_sha256 missing or empty")
            elif not exp_sha:
                problems.append("disk adapter meta hash missing")
            elif exp_sha != got_sha:
                problems.append(
                    f"adapter weight sha mismatch summary={got_sha} disk={exp_sha}"
                )
            ad = fp.get("adapter_dir")
            if not ad:
                problems.append("missing required fingerprint field: adapter_dir")
            else:
                try:
                    if Path(ad).resolve() != expected_adapter_dir.resolve():
                        if (
                            expected_adapter_dir.resolve().as_posix()
                            not in Path(ad).resolve().as_posix()
                        ):
                            problems.append(
                                f"adapter_dir fingerprint {ad} != {expected_adapter_dir}"
                            )
                except Exception:
                    problems.append(f"adapter_dir unresolvable: {ad}")
        tu = fp.get("train_update")
        if expected_train_update is not None:
            if tu is None or int(tu) != int(expected_train_update):
                problems.append(f"train_update={tu} expected={expected_train_update}")

    if problems:
        return "incompatible", f"{tag}: incompatible cached eval: " + "; ".join(problems)
    return (
        "ok_complete",
        f"{tag}: complete ok={n_ok} fail={n_fail} miss={n_miss}",
    )


def main() -> int:
    gate = P3_ART / "gate_ok.json"
    if not gate.is_file():
        print("ERROR: missing gate_ok.json — run preflight/gate first", file=sys.stderr)
        return 2
    g = json.loads(gate.read_text())
    if not g.get("ok"):
        print("ERROR: gate report ok!=true", file=sys.stderr)
        return 2

    for seed in SEEDS:
        adapter = seed_dirs(seed)["checkpoint_root"] / "update_0200"
        if adapter.is_dir() and (seed_dirs(seed)["summary_json"]).is_file():
            print(f"SKIP_TRAIN seed={seed} adapter exists", flush=True)
            continue
        rc = run_train(seed)
        if rc != 0:
            print(f"ERROR: train failed seed={seed} rc={rc}", file=sys.stderr)
            return rc

    base_dir = P3_ART / "eval_fresh_base"
    base_sum = base_dir / "dev_eval_summary.json"
    base_jsonl = base_dir / "dev_eval.jsonl"
    status, msg = validate_eval_cache(
        tag="fresh_base",
        summary_path=base_sum,
        jsonl_path=base_jsonl,
        expected_adapter_dir=None,
        expected_train_update=None,
    )
    print(f"EVAL_CACHE fresh_base status={status} {msg}", flush=True)
    if status == "incompatible":
        print(
            "ERROR: refusing to overwrite or auto-replace incompatible fresh_base cache",
            file=sys.stderr,
        )
        return 3
    if status == "missing":
        rc = run_eval(
            tag="fresh_base",
            lora_adapter=None,
            train_update=None,
            out_jsonl=base_jsonl,
            summary_json=base_sum,
        )
        if rc != 0:
            return rc
    else:
        print("SKIP_EVAL fresh_base (validated)", flush=True)

    for seed in SEEDS:
        d = seed_dirs(seed)
        adapter = d["checkpoint_root"] / "update_0200"
        status, msg = validate_eval_cache(
            tag=f"seed_{seed}",
            summary_path=d["eval_summary"],
            jsonl_path=d["eval_jsonl"],
            expected_adapter_dir=adapter,
            expected_train_update=200,
        )
        print(f"EVAL_CACHE seed_{seed} status={status} {msg}", flush=True)
        if status == "incompatible":
            print(
                f"ERROR: refusing to overwrite or auto-replace incompatible seed_{seed} cache",
                file=sys.stderr,
            )
            return 3
        if status == "ok_complete":
            print(f"SKIP_EVAL seed={seed} (validated)", flush=True)
            continue
        if not adapter.is_dir():
            print(f"ERROR: missing adapter {adapter}", file=sys.stderr)
            return 2
        rc = run_eval(
            tag=f"seed_{seed}",
            lora_adapter=str(adapter),
            train_update=200,
            out_jsonl=d["eval_jsonl"],
            summary_json=d["eval_summary"],
        )
        if rc != 0:
            return rc

    analyze()
    done = {
        "task": "pascal_p3_main_done",
        "seeds": list(SEEDS),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": os.uname().nodename,
        "repo_root": str(REPO_ROOT),
    }
    (P3_ART / "main_done.json").write_text(json.dumps(done, indent=2) + "\n")
    print("P3_MAIN_DONE", json.dumps(done), flush=True)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
