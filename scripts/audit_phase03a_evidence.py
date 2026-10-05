#!/usr/bin/env python3
"""Phase 3A evidence audit from actual train logs/predictions (no new training)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        print("ERROR: export PYTHONNOUSERSITE=1 before python", file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/phase03a_evidence_audit.json")
    parser.add_argument("--curve-svg", default="docs/examples/phase03a_train_loss_curve.svg")
    parser.add_argument("--changed-md", default="docs/examples/phase03a_eight_changed.md")
    args = parser.parse_args()

    root = _repo_root()
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))

    from vlm_doc.metrics import SCORER_VERSION_PRIMARY, score_example
    from vlm_doc.svg_plot import write_line_chart_svg

    train_jsonl = Path("results/train_internvl3_1b_lora_v1.jsonl")
    rows = [
        json.loads(l)
        for l in train_jsonl.read_text().splitlines()
        if l.strip()
    ]
    losses = [float(r["loss"]) for r in rows]
    updates = [int(r["update"]) for r in rows]
    assert updates == list(range(1, len(rows) + 1)), "update indices not contiguous 1..N"

    # Logged loss is mean over grad_accum microbatches within that optimizer update
    # (see scripts/train_internvl_lora.py mean_loss = sum(loss_window)/len(loss_window)).
    loss_semantics = {
        "field": "loss",
        "meaning": (
            "Per-optimizer-update mean of microbatch losses in the gradient-accumulation "
            "window (grad_accum=8). Not a single-example loss and not a sliding window "
            "across updates. First/final reported values are update-1 and update-N means "
            "over (possibly different) example sets."
        ),
        "grad_accum": 8,
        "n_updates": len(rows),
        "samples_seen_final": rows[-1]["samples_seen"],
        "unique_examples_seen_final": rows[-1]["unique_examples_seen"],
        "initial_loss_update_mean": losses[0],
        "final_loss_update_mean": losses[-1],
        "mean_loss_first_20_updates": sum(losses[:20]) / 20.0,
        "mean_loss_last_20_updates": sum(losses[-20:]) / 20.0,
        "label_note": (
            "Training statistics on potentially different examples across updates; "
            "not a held-out validation curve."
        ),
    }

    write_line_chart_svg(
        Path(args.curve_svg),
        xs=updates,
        ys=losses,
        title="Phase 3A InternVL LoRA train loss (per-update accum mean)",
        xlabel="optimizer update",
        ylabel="mean microbatch loss",
    )

    def load_preds(path: Path) -> dict[int, dict]:
        out: dict[int, dict] = {}
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            out[int(rec["question_id"])] = rec
        return out

    manifest = json.loads(Path("data/manifests/docvqa_dev_v1.json").read_text())
    req = sorted(int(e["question_id"]) for e in manifest["examples"])
    answers = json.loads(Path("data/cache/docvqa_dev_v1_answers.json").read_text())
    ans = {int(a["question_id"]): list(a["answers"]) for a in answers["answers"]}

    base = load_preds(Path("results/dev_internvl3_1b.jsonl"))
    u100 = load_preds(Path("results/dev_internvl3_1b_lora_u100.jsonl"))
    u200 = load_preds(Path("results/dev_internvl3_1b_lora_u200.jsonl"))

    coverage = {}
    for name, preds in [("step0", base), ("step100", u100), ("step200", u200)]:
        missing = [q for q in req if q not in preds]
        statuses = {}
        for q in req:
            if q in preds:
                statuses[preds[q].get("status", "?")] = (
                    statuses.get(preds[q].get("status", "?"), 0) + 1
                )
        coverage[name] = {
            "n_requested": len(req),
            "n_present": len([q for q in req if q in preds]),
            "missing_question_ids": missing,
            "status_counts": statuses,
            "all_accounted": not missing and statuses.get("ok", 0) == len(req),
        }

    expected_hashes = {
        100: "16564ec14d2d2a071f927534a84960ed43f0f0d194ec2d5fdc0bf7d48f67c219",
        200: "4b925a8528ce9fddbe3357db6d0223ea2b3547fb37c5380d0924d2bdbd36e1e7",
    }
    checkpoint_checks = {}
    for step, preds in [(100, u100), (200, u200)]:
        disk = Path(f"checkpoints/internvl3_1b_lora_v1/update_{step:04d}/adapter_model.safetensors")
        disk_sha = hashlib.sha256(disk.read_bytes()).hexdigest()
        fps = set()
        dirs = set()
        for rec in preds.values():
            fp = rec.get("run_fingerprint") or {}
            w = (fp.get("adapter_weights_sha256") or {}).get("adapter_model.safetensors")
            fps.add(w)
            dirs.add(fp.get("adapter_dir"))
        checkpoint_checks[step] = {
            "disk_adapter_sha256": disk_sha,
            "expected_sha256": expected_hashes[step],
            "disk_matches_expected": disk_sha == expected_hashes[step],
            "prediction_fingerprint_weight_shas": sorted(x for x in fps if x),
            "prediction_adapter_dirs": sorted(x for x in dirs if x),
            "preds_match_disk": fps == {disk_sha},
            "adapter_dir_ok": dirs
            == {f"checkpoints/internvl3_1b_lora_v1/update_{step:04d}"},
        }

    improved = []
    regressed = []
    tied = 0
    for qid in req:
        refs = ans[qid]
        bp = base[qid].get("prediction")
        ap = u100[qid].get("prediction")
        b = score_example(bp, refs, status="ok", primary=SCORER_VERSION_PRIMARY)
        a = score_example(ap, refs, status="ok", primary=SCORER_VERSION_PRIMARY)
        delta = float(a["anls"]) - float(b["anls"])
        row = {
            "question_id": qid,
            "question": base[qid].get("question"),
            "references": refs,
            "baseline_prediction": bp,
            "adapted_prediction_u100": ap,
            "baseline_anls_strict_v2": b["anls"],
            "adapted_anls_strict_v2": a["anls"],
            "delta_anls": delta,
        }
        if abs(delta) < 1e-12:
            tied += 1
        elif delta > 0:
            improved.append(row)
        else:
            regressed.append(row)

    changed = sorted(improved, key=lambda r: -r["delta_anls"]) + sorted(
        regressed, key=lambda r: r["delta_anls"]
    )

    # Gate discard evidence from source + job log
    gate_src = Path("scripts/gate_internvl_lora.py").read_text()
    sbatch = Path("scripts/slurm/phase03a_internvl_lora.sbatch").read_text()
    log_out = Path("logs/vlm_doc_lora3a_82985.out").read_text()
    discard_evidence = {
        "gate_uses_temp_dir_rmtree": "shutil.rmtree(tmp" in gate_src,
        "gate_docstring_requires_fresh_main_reload": (
            "Main training must reload" in gate_src
            or "reload fresh" in gate_src.lower()
        ),
        "sbatch_runs_gate_then_separate_train_process": (
            "gate_internvl_lora.py" in sbatch and "train_internvl_lora.py" in sbatch
        ),
        "job_log_has_main_train_after_gate_ok": (
            "GATE_OK" in log_out and "=== MAIN TRAIN 200 UPDATES ===" in log_out
        ),
        "interpretation": (
            "Memorization/save-reload adapters lived in a TemporaryDirectory removed "
            "at gate exit; main training is a separate Python process that attaches "
            "fresh LoRA to a newly loaded base. Diagnostic optimizer state does not "
            "persist across process boundaries."
        ),
    }

    md_lines = [
        "# Phase 3A — eight development examples that changed under LoRA u100",
        "",
        "Primary metric: `anls_normalized_strict_v2`. Comparison is baseline (step 0) vs "
        "best-trained checkpoint (update 100). Counts recomputed from prediction JSONLs.",
        "",
        f"Recomputed: **{len(improved)} improved** / **{len(regressed)} regressed** / "
        f"**{tied} tied** (expected 4 / 4 / 200).",
        "",
    ]
    for label, changed_rows in (("Improved", improved), ("Regressed", regressed)):
        md_lines.append(f"## {label}")
        md_lines.append("")
        for r in sorted(
            changed_rows,
            key=lambda x: (-x["delta_anls"] if label == "Improved" else x["delta_anls"]),
        ):
            md_lines.extend(
                [
                    f"### qid {r['question_id']}",
                    "",
                    f"- Question: {r['question']}",
                    f"- References: {r['references']!r}",
                    f"- Baseline prediction: {r['baseline_prediction']!r}",
                    f"- Adapted (u100) prediction: {r['adapted_prediction_u100']!r}",
                    f"- ANLS baseline → adapted: {r['baseline_anls_strict_v2']:.4f} → "
                    f"{r['adapted_anls_strict_v2']:.4f} (Δ {r['delta_anls']:+.4f})",
                    "",
                ]
            )
    Path(args.changed_md).write_text("\n".join(md_lines) + "\n")

    ok_parts = {
        "n_rows": len(rows) == 200,
        "c0": coverage["step0"]["all_accounted"],
        "c100": coverage["step100"]["all_accounted"],
        "c200": coverage["step200"]["all_accounted"],
        "disk": all(checkpoint_checks[s]["disk_matches_expected"] for s in (100, 200)),
        "preds": all(checkpoint_checks[s]["preds_match_disk"] for s in (100, 200)),
        "imp": len(improved) == 4,
        "reg": len(regressed) == 4,
        "tied": tied == 200,
        "rmtree": discard_evidence["gate_uses_temp_dir_rmtree"],
        "sbatch": discard_evidence["sbatch_runs_gate_then_separate_train_process"],
    }
    report = {
        "task": "phase03a_evidence_audit",
        "selection_freeze": {
            "deployment_candidate": "original InternVL3-1B (step 0)",
            "best_trained_checkpoint": "LoRA update 100",
            "update_200_status": "retained experiment result; not selected",
            "fine_tuning_improved_primary_dev_anls": False,
        },
        "train_curve": loss_semantics,
        "loss_curve_svg": args.curve_svg,
        "coverage": coverage,
        "checkpoint_checks": checkpoint_checks,
        "paired_anls_u100_vs_baseline": {
            "n_improved": len(improved),
            "n_regressed": len(regressed),
            "n_tied": tied,
            "matches_reported_4_4_200": (
                len(improved) == 4 and len(regressed) == 4 and tied == 200
            ),
            "changed_examples": changed,
        },
        "diagnostic_adapter_discard": discard_evidence,
        "observations_not_causal_claims": [
            "Primary development ANLS did not improve under this bounded LoRA recipe.",
            "Train loss decreased over 200 updates; that is a training statistic, not "
            "evidence of generalization or forgetting by itself.",
            "Do not treat the eight changed examples as proof of overfitting or "
            "catastrophic forgetting without further controlled analysis.",
        ],
        "ok_parts": ok_parts,
        "ok": all(ok_parts.values()),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"ok": report["ok"], "out": str(out), "curve": args.curve_svg}, indent=2))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
