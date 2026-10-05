#!/usr/bin/env python3
"""Build Phase 2B two-model development comparison from rescored JSONL/summaries."""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path


def _gib(n: int) -> float:
    return float(n) / (1024.0**3)


def _pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def load_last_by_qid(path: Path) -> dict[int, dict]:
    by: dict[int, dict] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            by[int(rec["question_id"])] = rec
    return by


def main() -> int:
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        print("ERROR: PYTHONNOUSERSITE=1 required", file=sys.stderr)
        return 2
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))

    from vlm_doc.metrics import (
        anls_normalized_inclusive_v1,
        anls_normalized_strict_v2,
        anls_vlmevalkit_single,
        exact_match_single,
    )

    manifest = json.loads(Path("data/manifests/docvqa_dev_v1.json").read_text())
    qids = [int(x) for x in manifest["question_ids"]]
    answers = {
        int(a["question_id"]): a["answers"]
        for a in json.loads(Path("data/cache/docvqa_dev_v1_answers.json").read_text())[
            "answers"
        ]
    }

    m256 = load_last_by_qid(Path("results/dev_smolvlm256m.jsonl"))
    m500 = load_last_by_qid(Path("results/dev_smolvlm500m.jsonl"))
    s500 = json.loads(Path("results/dev_smolvlm500m_summary.json").read_text())
    s256_hist = json.loads(
        Path("results/historical/dev_smolvlm256m_summary_phase02a.json").read_text()
    )
    s256_v2 = json.loads(
        Path("results/dev_smolvlm256m_rescore_strict_v2.json").read_text()
    )

    def score_model(by: dict[int, dict]) -> dict:
        strict = em = incl = vlk = 0.0
        lats: list[float] = []
        lens: list[int] = []
        n_cap = 0
        n_ok = 0
        peak_a = peak_r = 0
        gpus: set[str] = set()
        for qid in qids:
            rec = by.get(qid)
            if rec is None or rec.get("status") != "ok":
                continue
            n_ok += 1
            pred = rec.get("prediction") or ""
            refs = answers[qid]
            strict += anls_normalized_strict_v2(pred, refs)
            incl += anls_normalized_inclusive_v1(pred, refs)
            vlk += anls_vlmevalkit_single(pred, refs)
            em += exact_match_single(pred, refs)
            if rec.get("generation_seconds") is not None:
                lats.append(float(rec["generation_seconds"]))
            if rec.get("generated_token_length") is not None:
                gl = int(rec["generated_token_length"])
                lens.append(gl)
                if gl >= int(rec.get("max_new_tokens") or 64):
                    n_cap += 1
            gm = rec.get("gpu_memory") or {}
            peak_a = max(peak_a, int(gm.get("peak_allocated_bytes") or 0))
            peak_r = max(peak_r, int(gm.get("peak_reserved_bytes") or 0))
            hw = rec.get("hardware") or {}
            if hw.get("gpu_name"):
                gpus.add(hw["gpu_name"])
        n = len(qids)
        return {
            "n_requested": n,
            "n_ok": n_ok,
            "n_failed": n - n_ok,
            "anls_strict_v2": strict / n,
            "anls_inclusive_v1": incl / n,
            "anls_vlmevalkit": vlk / n,
            "exact_match": em / n,
            "latency_median": statistics.median(lats) if lats else None,
            "latency_p95": _pct(lats, 95),
            "median_gen_tokens": statistics.median(lens) if lens else None,
            "n_generation_cap_reached": n_cap,
            "peak_allocated_gib": _gib(peak_a),
            "peak_reserved_gib": _gib(peak_r),
            "gpus": sorted(gpus),
        }

    c256 = score_model(m256)
    c500 = score_model(m500)

    # Per-question deltas on strict v2
    improved = []
    regressed = []
    tied = 0
    for qid in qids:
        a = m256.get(qid)
        b = m500.get(qid)
        if not a or not b or a.get("status") != "ok" or b.get("status") != "ok":
            continue
        sa = anls_normalized_strict_v2(a["prediction"], answers[qid])
        sb = anls_normalized_strict_v2(b["prediction"], answers[qid])
        d = sb - sa
        row = {
            "question_id": qid,
            "delta": d,
            "anls_256": sa,
            "anls_500": sb,
            "pred_256": a["prediction"],
            "pred_500": b["prediction"],
            "refs": answers[qid],
            "question": a.get("question") or b.get("question"),
        }
        if d > 1e-12:
            improved.append(row)
        elif d < -1e-12:
            regressed.append(row)
        else:
            tied += 1
    improved.sort(key=lambda r: (-r["delta"], r["question_id"]))
    regressed.sort(key=lambda r: (r["delta"], r["question_id"]))

    gallery_qids = [5158, 292, 297, 5263, 53842]  # development only
    paired = []
    for qid in gallery_qids:
        a = m256[qid]
        b = m500[qid]
        refs = answers[qid]
        paired.append(
            {
                "question_id": qid,
                "question": a["question"],
                "refs": refs,
                "pred_256": a["prediction"],
                "pred_500": b["prediction"],
                "anls_256_strict": anls_normalized_strict_v2(a["prediction"], refs),
                "anls_500_strict": anls_normalized_strict_v2(b["prediction"], refs),
                "em_256": exact_match_single(a["prediction"], refs),
                "em_500": exact_match_single(b["prediction"], refs),
            }
        )

    out = {
        "primary_metric": "anls_normalized_strict_v2",
        "n_development": len(qids),
        "smolvlm_256m": {
            **c256,
            "model_revision": s256_hist.get("model_revision"),
            "total_unique_parameters": s256_hist.get("total_unique_parameters"),
            "job_id": (s256_hist.get("run_metadata") or {}).get("slurm_job_id"),
            "rescore_note": "CPU rescored under strict_v2; original inference provenance retained",
            "historical_inclusive_mean": s256_hist.get("scores", {}).get(
                "mean_anls_requested"
            ),
            "cpu_rescore_strict_mean": s256_v2["scores"]["mean_anls_requested"],
            "n_cutoff_diff_vs_inclusive": s256_v2[
                "n_examples_cutoff_diff_strict_vs_inclusive"
            ],
        },
        "smolvlm_500m": {
            **c500,
            "model_revision": s500.get("model_revision"),
            "total_unique_parameters": s500.get("total_unique_parameters"),
            "job_id": (s500.get("run_metadata") or {}).get("slurm_job_id"),
            "dtype": s500.get("dtype"),
            "processor_settings": s500.get("processor_settings"),
            "hardware_summary": s500.get("hardware"),
        },
        "per_question": {
            "n_improved_500_vs_256": len(improved),
            "n_regressed_500_vs_256": len(regressed),
            "n_tied": tied,
            "note": (
                "Point estimates only; no statistical significance claimed from "
                "these counts alone."
            ),
            "top_improvements": improved[:10],
            "top_regressions": regressed[:10],
        },
        "gallery_paired_development": paired,
    }
    Path("results/dev_256_vs_500_comparison.json").write_text(
        json.dumps(out, indent=2) + "\n"
    )
    print(json.dumps({
        "256_strict": c256["anls_strict_v2"],
        "500_strict": c500["anls_strict_v2"],
        "256_em": c256["exact_match"],
        "500_em": c500["exact_match"],
        "improved": len(improved),
        "regressed": len(regressed),
        "tied": tied,
        "500_params": s500.get("total_unique_parameters"),
        "500_gpu": c500["gpus"],
        "500_med_lat": c500["latency_median"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
