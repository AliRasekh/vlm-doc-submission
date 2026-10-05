#!/usr/bin/env python3
"""Three-model development comparison under strict_v2 (+ diagnostic EM)."""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path


def load_last(path: Path) -> dict[int, dict]:
    by: dict[int, dict] = {}
    with path.open() as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                by[int(r["question_id"])] = r
    return by


def gib(n: int) -> float:
    return n / (1024.0**3)


def pct(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


def main() -> int:
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        return 2
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))
    from vlm_doc.metrics import (
        anls_normalized_strict_v2,
        exact_match_single,
        normalize_text,
    )

    def diag_em(pred, refs):
        def strip_dot(n):
            return n[:-1] if n.endswith(".") else n

        p = strip_dot(normalize_text(pred))
        for r in refs:
            rn = normalize_text(r)
            if not rn:
                continue
            if p == strip_dot(rn):
                return 1.0
        return 0.0

    man = json.loads(Path("data/manifests/docvqa_dev_v1.json").read_text())
    qids = [int(x) for x in man["question_ids"]]
    ans = {
        int(a["question_id"]): a["answers"]
        for a in json.loads(Path("data/cache/docvqa_dev_v1_answers.json").read_text())[
            "answers"
        ]
    }
    models = {
        "smolvlm_256m": {
            "jsonl": Path("results/dev_smolvlm256m.jsonl"),
            "summary": Path("results/historical/dev_smolvlm256m_summary_phase02a.json"),
        },
        "smolvlm_500m": {
            "jsonl": Path("results/dev_smolvlm500m.jsonl"),
            "summary": Path("results/dev_smolvlm500m_summary.json"),
        },
        "internvl3_1b": {
            "jsonl": Path("results/dev_internvl3_1b.jsonl"),
            "summary": Path("results/dev_internvl3_1b_summary.json"),
        },
    }
    out_models = {}
    preds = {}
    for name, paths in models.items():
        by = load_last(paths["jsonl"])
        assert set(by) == set(qids)
        summary = json.loads(paths["summary"].read_text())
        anls = em = dem = 0.0
        lats = []
        lenses = []
        n_cap = 0
        n_cap_unavail = 0
        peak_a = peak_r = 0
        tiles = []
        vtoks = []
        for qid in qids:
            r = by[qid]
            pred = r.get("prediction") or ""
            refs = ans[qid]
            anls += anls_normalized_strict_v2(pred, refs)
            em += exact_match_single(pred, refs)
            dem += diag_em(pred, refs)
            if r.get("generation_seconds") is not None:
                lats.append(float(r["generation_seconds"]))
            gl = r.get("generated_token_length")
            if gl is not None:
                lenses.append(int(gl))
            cap = r.get("generation_cap_reached")
            if cap is None:
                n_cap_unavail += 1
            elif cap:
                n_cap += 1
            gm = r.get("gpu_memory") or {}
            peak_a = max(peak_a, int(gm.get("peak_allocated_bytes") or 0))
            peak_r = max(peak_r, int(gm.get("peak_reserved_bytes") or 0))
            ii = r.get("image_info") or {}
            if "tile_count" in ii:
                tiles.append(int(ii["tile_count"]))
            if "visual_token_count" in ii:
                vtoks.append(int(ii["visual_token_count"]))
        out_models[name] = {
            "model_id": summary.get("model_id"),
            "model_revision": summary.get("model_revision"),
            "total_unique_parameters": summary.get("total_unique_parameters"),
            "n_requested": 208,
            "n_ok": sum(1 for q in qids if by[q].get("status") == "ok"),
            "anls_strict_v2": anls / 208,
            "exact_match": em / 208,
            "diagnostic_terminal_period_em": dem / 208,
            "latency_median": statistics.median(lats) if lats else None,
            "latency_p95": pct(lats, 95),
            "timing_scope": summary.get("latency_seconds", {}).get(
                "hardware_timing_label"
            )
            or (by[qids[0]].get("timing_scope") if by else None),
            "hardware": summary.get("hardware"),
            "peak_allocated_gib": gib(peak_a),
            "peak_reserved_gib": gib(peak_r),
            "gen_token_median": statistics.median(lenses) if lenses else None,
            "n_generation_cap_reached": n_cap,
            "n_generation_cap_unavailable": n_cap_unavail,
            "tile_count_median": statistics.median(tiles) if tiles else None,
            "visual_token_count_median": statistics.median(vtoks) if vtoks else None,
            "processor_settings": summary.get("processor_settings"),
            "adapter": summary.get("adapter"),
            "job_id": (summary.get("run_metadata") or {}).get("slurm_job_id"),
        }
        preds[name] = by

    gallery = []
    for qid in [5158, 292, 297, 5263, 53842]:
        row = {
            "question_id": qid,
            "question": man["examples"][
                next(i for i, e in enumerate(man["examples"]) if int(e["question_id"]) == qid)
            ]["question"],
            "refs": ans[qid],
        }
        for name in models:
            p = preds[name][qid]["prediction"]
            row[f"pred_{name}"] = p
            row[f"anls_{name}"] = anls_normalized_strict_v2(p, ans[qid])
            row[f"em_{name}"] = exact_match_single(p, ans[qid])
        gallery.append(row)

    report = {
        "primary_metric": "anls_normalized_strict_v2",
        "diagnostic_metric": "terminal_period_em_after_normalize",
        "note": (
            "Development ranking is provisional and does not constitute final "
            "holdout performance. Diagnostic EM is secondary."
        ),
        "models": out_models,
        "gallery_paired_development": gallery,
    }
    Path("results/dev_three_model_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps({k: {
        "anls": v["anls_strict_v2"],
        "em": v["exact_match"],
        "diag_em": v["diagnostic_terminal_period_em"],
        "params": v["total_unique_parameters"],
        "med_lat": v["latency_median"],
    } for k, v in out_models.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
