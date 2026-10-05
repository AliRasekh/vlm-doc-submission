#!/usr/bin/env python3
"""Compare InternVL baseline vs LoRA checkpoints on development; select by ANLS."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_jsonl(path: Path) -> dict[int, dict]:
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[int(r["question_id"])] = r
    return out


def anls_of(r: dict) -> float:
    m = r.get("metrics") or {}
    v = m.get("anls_normalized_strict_v2")
    if v is None:
        v = r.get("anls_normalized_strict_v2")
    return float(v or 0.0)


def em_of(r: dict) -> float:
    m = r.get("metrics") or {}
    v = m.get("exact_match")
    if v is None:
        v = r.get("exact_match")
    return float(v or 0.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="results/dev_internvl3_1b.jsonl")
    parser.add_argument("--step100", required=True)
    parser.add_argument("--step200", required=True)
    parser.add_argument("--answers", default="data/cache/docvqa_dev_v1_answers.json")
    parser.add_argument("--out", default="results/dev_internvl_lora_v1_comparison.json")
    parser.add_argument("--examples-md", default="docs/examples/lora_v1_before_after.md")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base = load_jsonl(Path(args.baseline))
    s100 = load_jsonl(Path(args.step100))
    s200 = load_jsonl(Path(args.step200))
    ans = json.loads(Path(args.answers).read_text())
    refs = {int(r["question_id"]): r["answers"] for r in ans["answers"]}

    qids = sorted(set(base) & set(s100) & set(s200))
    assert len(qids) == 208, f"expected 208 paired qids, got {len(qids)}"

    def mean_anls(d):
        return sum(anls_of(d[q]) for q in qids) / len(qids)

    def mean_em(d):
        return sum(em_of(d[q]) for q in qids) / len(qids)

    scores = {
        0: {"anls": mean_anls(base), "em": mean_em(base), "label": "step0_baseline"},
        100: {"anls": mean_anls(s100), "em": mean_em(s100), "label": "step100"},
        200: {"anls": mean_anls(s200), "em": mean_em(s200), "label": "step200"},
    }

    # Prefer earlier step on exact ANLS ties.
    ordered = sorted(
        scores.items(),
        key=lambda kv: (-kv[1]["anls"], kv[0]),
    )
    selected_step = ordered[0][0]
    best_trained = max(
        [(100, scores[100]["anls"]), (200, scores[200]["anls"])],
        key=lambda x: (x[1], -x[0]),
    )[0]

    # Example buckets: always compare best trained checkpoint vs baseline
    # (even when selection retains step 0).
    adapted = s100 if best_trained == 100 else s200
    improved, regressed, unchanged = [], [], []
    for q in qids:
        d_sel = anls_of(adapted[q])
        delta = d_sel - anls_of(base[q])
        row = {
            "question_id": q,
            "question": base[q].get("question"),
            "baseline_pred": base[q].get("prediction"),
            "baseline_prediction_raw": base[q].get("prediction_raw"),
            "baseline_prediction_decoded_full": base[q].get("prediction_decoded_full"),
            "adapted_pred": adapted[q].get("prediction"),
            "adapted_prediction_raw": adapted[q].get("prediction_raw"),
            "adapted_prediction_decoded_full": adapted[q].get(
                "prediction_decoded_full"
            ),
            "references": refs.get(q),
            "baseline_anls": anls_of(base[q]),
            "adapted_anls": d_sel,
            "delta_anls": delta,
            "adapted_checkpoint_update": best_trained,
        }
        if delta > 1e-12:
            improved.append(row)
        elif delta < -1e-12:
            regressed.append(row)
        else:
            unchanged.append(row)

    def take(xs, n=3):
        xs = sorted(
            xs, key=lambda r: (abs(r["delta_anls"]), r["question_id"]), reverse=True
        )
        return xs[:n]

    examples = {
        "improvements": take(improved, 3),
        "regressions": take(regressed, 3),
        "unchanged": take(unchanged, 3),
    }

    report = {
        "task": "dev_internvl_lora_v1_comparison",
        "n_paired": len(qids),
        "scores": {str(k): v for k, v in scores.items()},
        "selection_rule": (
            "Compare step 0/100/200 primary development ANLS; "
            "prefer earlier step on exact ties. Also name best trained among 100/200."
        ),
        "selected_step_for_dev": selected_step,
        "selected_label": scores[selected_step]["label"],
        "best_trained_step": best_trained,
        "examples_compare_checkpoint": best_trained,
        "fine_tuning_improved_selection": selected_step != 0,
        "n_improved_vs_baseline_on_best_trained": len(improved),
        "n_regressed_vs_baseline_on_best_trained": len(regressed),
        "n_unchanged_vs_baseline_on_best_trained": len(unchanged),
        "examples": examples,
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# InternVL LoRA before/after development examples (Phase 3A)",
        "",
        f"Selected step (dev selection): **{selected_step}** (`{scores[selected_step]['label']}`). "
        f"Best trained among 100/200: **{best_trained}** (examples below vs baseline).",
        "",
        f"| Step | ANLS strict v2 | EM |",
        f"|-----:|---------------:|---:|",
        f"| 0 (baseline) | {scores[0]['anls']:.4f} | {scores[0]['em']:.4f} |",
        f"| 100 | {scores[100]['anls']:.4f} | {scores[100]['em']:.4f} |",
        f"| 200 | {scores[200]['anls']:.4f} | {scores[200]['em']:.4f} |",
        "",
        f"Counts vs baseline at update {best_trained}: "
        f"improved={len(improved)}, regressed={len(regressed)}, unchanged={len(unchanged)}.",
        "",
    ]
    for kind, rows in examples.items():
        lines.append(f"## {kind}")
        lines.append("")
        for r in rows:
            lines.append(f"### qid {r['question_id']}")
            lines.append("")
            lines.append(f"- Q: {r['question']}")
            lines.append(f"- Refs: {r['references']}")
            lines.append(f"- Baseline pred: {r['baseline_pred']!r} (ANLS {r['baseline_anls']:.4f})")
            lines.append(f"- Adapted pred: {r['adapted_pred']!r} (ANLS {r['adapted_anls']:.4f})")
            lines.append(f"- ΔANLS: {r['delta_anls']:+.4f}")
            lines.append("")
    Path(args.examples_md).write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "selected_step": selected_step,
        "best_trained_step": best_trained,
        "scores": {str(k): v["anls"] for k, v in scores.items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
