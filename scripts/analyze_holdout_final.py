#!/usr/bin/env python3
"""Final holdout analysis: tables, paired deltas, diagnostic EM, gallery, figures."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_jsonl(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[int(rec["question_id"])] = rec
    return out


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def count_ucsf_groups(manifest_examples: list[dict]) -> int:
    groups: set[str] = set()
    for ex in manifest_examples:
        qid = int(ex["question_id"])
        groups.add(str(ex.get("ucsf_document_id") or f"MISSING::{qid}"))
    return len(groups)


def mean_requested_anls(preds: dict[int, dict], req: list[int]) -> float:
    """Requested-set mean primary ANLS; non-ok / missing contribute 0."""
    total = 0.0
    for q in req:
        rec = preds.get(q)
        if not rec or rec.get("status") != "ok":
            continue
        m = rec.get("metrics") or {}
        total += float(m.get("anls_normalized_strict_v2", m.get("anls", 0.0)))
    return total / max(len(req), 1)


def attach_bootstrap_ci(
    *,
    bootstrap_path: str | None,
    bootstrap_obj: dict | None,
    manifest_examples: list[dict],
    base_preds: dict[int, dict],
    adapted_preds: dict[int, dict],
    recomputed_base_mean: float,
    recomputed_adapted_mean: float,
) -> dict:
    """Attach CI only when an explicit bootstrap artifact corresponds to this analysis.

    Correspondence uses available sealed provenance fields: task/metric labels,
    exact n_questions / n_groups vs the analyzed manifest, and exact (unrounded)
    point means / difference vs means recomputed from the analyzed prediction
    records. Matching rounded scores alone is rejected. Prediction-file content
    hashes are not present in the sealed bootstrap artifact, so exact point
    statistics + manifest structure are the binding checks available here.
    """
    if not bootstrap_path:
        return {
            "bootstrap": {},
            "bootstrap_attached": False,
            "bootstrap_omit_reason": (
                "No --bootstrap-json provided; refusing to auto-load "
                "results/holdout_internvl_anls_bootstrap.json for a possibly "
                "unrelated prediction set."
            ),
        }
    if bootstrap_obj is None:
        return {
            "bootstrap": {},
            "bootstrap_attached": False,
            "bootstrap_omit_reason": f"Bootstrap path not readable: {bootstrap_path}",
        }

    req = [int(e["question_id"]) for e in manifest_examples]
    n_groups = count_ucsf_groups(manifest_examples)
    point_diff = recomputed_adapted_mean - recomputed_base_mean
    reasons: list[str] = []

    if bootstrap_obj.get("task") != "holdout_ucsf_cluster_bootstrap_anls_diff":
        reasons.append(
            f"unexpected task={bootstrap_obj.get('task')!r} "
            "(expected holdout_ucsf_cluster_bootstrap_anls_diff)"
        )
    if bootstrap_obj.get("primary_metric") not in (
        None,
        "anls_normalized_strict_v2",
    ):
        # Accept missing primary_metric only if other checks pass? Prefer require match when present.
        if bootstrap_obj.get("primary_metric") != "anls_normalized_strict_v2":
            reasons.append(
                f"primary_metric={bootstrap_obj.get('primary_metric')!r} "
                "does not match anls_normalized_strict_v2"
            )
    if int(bootstrap_obj.get("n_questions") or -1) != len(req):
        reasons.append(
            f"n_questions={bootstrap_obj.get('n_questions')} != manifest n={len(req)}"
        )
    if int(bootstrap_obj.get("n_groups") or -1) != n_groups:
        reasons.append(
            f"n_groups={bootstrap_obj.get('n_groups')} != manifest UCSF groups={n_groups}"
        )

    # Exact float equality (not rounded display values).
    for key, got in (
        ("point_mean_anls_base", recomputed_base_mean),
        ("point_mean_anls_adapted", recomputed_adapted_mean),
        ("point_diff_adapted_minus_base", point_diff),
    ):
        if key not in bootstrap_obj:
            reasons.append(f"bootstrap missing provenance field {key}")
            continue
        stored = float(bootstrap_obj[key])
        if stored != got:
            reasons.append(
                f"{key} mismatch: bootstrap={stored!r} vs recomputed_from_analyzed_preds={got!r}"
            )

    # Coverage: every manifest QID must exist in both prediction maps used for the means.
    missing_base = [q for q in req if q not in base_preds]
    missing_adapted = [q for q in req if q not in adapted_preds]
    if missing_base or missing_adapted:
        reasons.append(
            "analyzed prediction maps incomplete vs manifest "
            f"(missing_base={len(missing_base)}, missing_adapted={len(missing_adapted)})"
        )

    if reasons:
        return {
            "bootstrap": {},
            "bootstrap_attached": False,
            "bootstrap_path": bootstrap_path,
            "bootstrap_omit_reason": (
                "Explicit bootstrap JSON failed correspondence checks against "
                "the analyzed predictions/manifest; CI omitted. " + "; ".join(reasons)
            ),
        }
    return {
        "bootstrap": bootstrap_obj,
        "bootstrap_attached": True,
        "bootstrap_path": bootstrap_path,
        "bootstrap_omit_reason": None,
        "bootstrap_correspondence": {
            "checked": [
                "task",
                "primary_metric",
                "n_questions",
                "n_groups",
                "point_mean_anls_base",
                "point_mean_anls_adapted",
                "point_diff_adapted_minus_base",
                "prediction_coverage_vs_manifest",
            ],
            "note": (
                "Sealed bootstrap artifact does not store prediction JSONL content "
                "hashes; correspondence uses exact point statistics and manifest "
                "structure available in the artifact."
            ),
        },
    }


def main() -> int:
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        print("ERROR: export PYTHONNOUSERSITE=1 before python", file=sys.stderr)
        return 2
    ap = argparse.ArgumentParser(
        description=(
            "Recompute holdout tables/analysis from saved prediction JSONL + summaries. "
            "This does not run model inference. To display sealed compact results without "
            "rescanning predictions, open results/holdout_final_table.json directly."
        )
    )
    ap.add_argument("--out", default="results/holdout_final_analysis.json")
    ap.add_argument("--gallery-md", default="docs/report/holdout_gallery.md")
    ap.add_argument("--table-json", default="results/holdout_final_table.json")
    ap.add_argument(
        "--smolvlm256m-jsonl",
        default="results/holdout_smolvlm256m.jsonl",
    )
    ap.add_argument(
        "--smolvlm256m-summary",
        default="results/holdout_smolvlm256m_summary.json",
    )
    ap.add_argument(
        "--smolvlm500m-jsonl",
        default="results/holdout_smolvlm500m.jsonl",
    )
    ap.add_argument(
        "--smolvlm500m-summary",
        default="results/holdout_smolvlm500m_summary.json",
    )
    ap.add_argument(
        "--internvl-jsonl",
        default="results/holdout_internvl3_1b.jsonl",
        help="Official sealed path; submission guide eval_* names are aliases for fresh runs.",
    )
    ap.add_argument(
        "--internvl-summary",
        default="results/holdout_internvl3_1b_summary.json",
    )
    ap.add_argument(
        "--internvl-lora-jsonl",
        default="results/holdout_internvl3_1b_lora_v2_u200.jsonl",
    )
    ap.add_argument(
        "--internvl-lora-summary",
        default="results/holdout_internvl3_1b_lora_v2_u200_summary.json",
    )
    ap.add_argument(
        "--lora-adapter-dir",
        default="artifacts/internvl3_1b_lora_v2_u200",
        help="Metadata reference only (not loaded here).",
    )
    ap.add_argument(
        "--figure-svg",
        default="results/holdout_primary_anls_recompute.svg",
        help=(
            "Output path for the primary-ANLS bar chart. Defaults under results/ so "
            "recomputes do not overwrite sealed docs/report/figures/ assets."
        ),
    )
    ap.add_argument(
        "--bootstrap-json",
        default=None,
        help=(
            "Optional path to a UCSF-cluster bootstrap JSON. Never auto-loaded. "
            "Attached only after exact correspondence checks against the analyzed "
            "predictions and manifest; otherwise the CI is omitted with an explanation."
        ),
    )
    ap.add_argument(
        "--manifest",
        default="data/manifests/docvqa_holdout_v2.json",
    )
    ap.add_argument(
        "--answers",
        default="data/cache/docvqa_holdout_v2_answers.json",
    )
    args = ap.parse_args()

    root = _repo_root()
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))

    from vlm_doc.metrics import (
        SCORER_VERSION_PRIMARY,
        exact_match_single,
        normalize_text,
        score_example,
    )
    from vlm_doc.svg_plot import write_grouped_bar_svg

    def strip_one_terminal_period(normalized: str) -> str:
        if normalized.endswith("."):
            return normalized[:-1]
        return normalized

    def diagnostic_em(pred: str | None, refs: list[str], status: str) -> float:
        if status != "ok" or pred is None:
            return 0.0
        pred_n = strip_one_terminal_period(normalize_text(pred))
        for r in refs:
            if strip_one_terminal_period(normalize_text(r)) == pred_n:
                return 1.0
        return 0.0

    systems = [
        ("smolvlm256m", args.smolvlm256m_jsonl, args.smolvlm256m_summary, None),
        ("smolvlm500m", args.smolvlm500m_jsonl, args.smolvlm500m_summary, None),
        ("internvl3_1b", args.internvl_jsonl, args.internvl_summary, None),
        (
            "internvl3_1b_lora_v2_u200",
            args.internvl_lora_jsonl,
            args.internvl_lora_summary,
            args.lora_adapter_dir,
        ),
    ]

    manifest = json.loads(Path(args.manifest).read_text())
    req = [int(e["question_id"]) for e in manifest["examples"]]
    q_by_id = {int(e["question_id"]): e for e in manifest["examples"]}
    answers = {
        int(a["question_id"]): list(a["answers"])
        for a in json.loads(Path(args.answers).read_text())["answers"]
    }

    rows = []
    preds = {}
    for tag, jpath, spath, adapter in systems:
        summary = json.loads(Path(spath).read_text())
        pred = _load_jsonl(Path(jpath))
        preds[tag] = pred
        missing = [q for q in req if q not in pred]
        n_ok = sum(1 for q in req if pred.get(q, {}).get("status") == "ok")
        n_fail = sum(
            1 for q in req if q in pred and pred[q].get("status") != "ok"
        )
        # Rescore secondary diagnostic EM on requested set
        diag_sum = 0.0
        vlk_sum = 0.0
        anls_sum = 0.0
        em_sum = 0.0
        latencies = []
        n_cap = 0
        for q in req:
            rec = pred.get(q)
            refs = answers[q]
            if rec is None:
                continue
            st = rec.get("status", "error")
            m = rec.get("metrics") or score_example(
                rec.get("prediction"), refs, status=st, primary=SCORER_VERSION_PRIMARY
            )
            anls_sum += float(m.get("anls_normalized_strict_v2", m.get("anls", 0.0)))
            em_sum += float(m.get("exact_match", 0.0))
            vlk_sum += float(m.get("anls_vlmevalkit", 0.0))
            diag_sum += diagnostic_em(rec.get("prediction"), refs, st)
            if rec.get("generation_cap_reached"):
                n_cap += 1
            if st == "ok" and rec.get("generation_seconds") is not None:
                latencies.append(float(rec["generation_seconds"]))
        n_req = len(req)
        scores = summary.get("scores") or {}
        hw = summary.get("hardware") or {}
        gpu_mem = summary.get("gpu_memory_peaks") or {}
        lat = summary.get("latency_seconds") or {}
        rows.append(
            {
                "system": tag,
                "adapter_dir": adapter,
                "total_unique_parameters": summary.get("total_unique_parameters"),
                "primary_anls_strict_v2": anls_sum / n_req,
                "primary_em": em_sum / n_req,
                "secondary_anls_vlmevalkit": vlk_sum / n_req,
                "secondary_diagnostic_terminal_period_em": diag_sum / n_req,
                "n_requested": n_req,
                "n_ok": n_ok,
                "n_failed": n_fail,
                "n_missing": len(missing),
                "n_generation_cap_hits": n_cap,
                "latency_generation_only_median_s": lat.get("median")
                if lat.get("median") is not None
                else (statistics.median(latencies) if latencies else None),
                "latency_generation_only_p95_s": lat.get("p95")
                if lat.get("p95") is not None
                else _percentile(latencies, 95),
                "latency_warmup_excluded": True,
                "latency_scope": "model.generate CUDA-synchronized",
                "peak_allocated_bytes": gpu_mem.get("peak_allocated_bytes"),
                "peak_reserved_bytes": gpu_mem.get("peak_reserved_bytes"),
                "gpu_memory_scope": gpu_mem.get("scope"),
                "gpu_name": (
                    hw.get("gpu_name")
                    or (
                        (hw.get("gpus") or [None])[0]
                        if isinstance(hw.get("gpus"), list)
                        else None
                    )
                ),
                "summary_path": spath,
                "jsonl_path": jpath,
                "run_fingerprint_sha256": (summary.get("run_fingerprint") or {}).get(
                    "run_fingerprint_sha256"
                ),
                "adapter_weights_sha256": (summary.get("run_fingerprint") or {}).get(
                    "adapter_weights_sha256"
                ),
                "scores_from_summary": {
                    "mean_anls_requested": scores.get("mean_anls_requested"),
                    "mean_exact_match_requested": scores.get(
                        "mean_exact_match_requested"
                    ),
                    "mean_anls_vlmevalkit_requested": scores.get(
                        "mean_anls_vlmevalkit_requested"
                    ),
                },
            }
        )

    # Paired InternVL base vs adapted
    base = preds["internvl3_1b"]
    adapted = preds["internvl3_1b_lora_v2_u200"]
    improved = []
    regressed = []
    tied = 0
    deltas = []
    for q in req:
        refs = answers[q]
        bp = base[q].get("prediction")
        ap = adapted[q].get("prediction")
        bm = score_example(bp, refs, status=base[q].get("status", "error"))
        am = score_example(ap, refs, status=adapted[q].get("status", "error"))
        d = float(am["anls"]) - float(bm["anls"])
        deltas.append(d)
        row = {
            "question_id": q,
            "question": q_by_id[q].get("question"),
            "references": refs,
            "baseline_pred": bp,
            "adapted_pred": ap,
            "baseline_anls": bm["anls"],
            "adapted_anls": am["anls"],
            "delta_anls": d,
            "ucsf_document_id": q_by_id[q].get("ucsf_document_id"),
            "image_relpath": q_by_id[q].get("image_relpath"),
        }
        if abs(d) < 1e-12:
            tied += 1
        elif d > 0:
            improved.append(row)
        else:
            regressed.append(row)

    recomputed_base = mean_requested_anls(base, req)
    recomputed_adapted = mean_requested_anls(adapted, req)
    boot_obj = None
    if args.bootstrap_json:
        bp = Path(args.bootstrap_json)
        if bp.is_file():
            boot_obj = json.loads(bp.read_text())
    boot_attach = attach_bootstrap_ci(
        bootstrap_path=args.bootstrap_json,
        bootstrap_obj=boot_obj,
        manifest_examples=manifest["examples"],
        base_preds=base,
        adapted_preds=adapted,
        recomputed_base_mean=recomputed_base,
        recomputed_adapted_mean=recomputed_adapted,
    )
    boot = boot_attach["bootstrap"]

    # Gallery: deterministic categories
    def first_exact_success():
        for q in sorted(req):
            refs = answers[q]
            if exact_match_single(base[q].get("prediction") or "", refs) == 1.0:
                return q
        return None

    def first_substantive_error():
        # ANLS=0 and not empty pred
        for q in sorted(req):
            m = score_example(
                base[q].get("prediction"),
                answers[q],
                status=base[q].get("status", "error"),
            )
            if m["anls"] == 0.0 and (base[q].get("prediction") or "").strip():
                return q
        return None

    gallery = {
        "selection_rules": {
            "exact_success": "Smallest qid where base primary EM=1",
            "substantive_error": "Smallest qid where base ANLS=0 and prediction nonempty",
            "adaptation_improvement": "Largest +Δ ANLS (adapted−base), tie→smallest qid",
            "adaptation_regression": "Most negative Δ ANLS, tie→smallest qid",
        },
        "categories": {},
    }
    q_success = first_exact_success()
    q_err = first_substantive_error()
    q_imp = sorted(improved, key=lambda r: (-r["delta_anls"], r["question_id"]))[0][
        "question_id"
    ] if improved else None
    q_reg = sorted(regressed, key=lambda r: (r["delta_anls"], r["question_id"]))[0][
        "question_id"
    ] if regressed else None
    for name, qid in [
        ("exact_success", q_success),
        ("substantive_error", q_err),
        ("adaptation_improvement", q_imp),
        ("adaptation_regression", q_reg),
    ]:
        if qid is None:
            gallery["categories"][name] = None
            continue
        gallery["categories"][name] = {
            "question_id": qid,
            "question": q_by_id[qid].get("question"),
            "references": answers[qid],
            "image_relpath": q_by_id[qid].get("image_relpath"),
            "base_prediction": base[qid].get("prediction"),
            "adapted_prediction": adapted[qid].get("prediction"),
            "base_anls": score_example(
                base[qid].get("prediction"),
                answers[qid],
                status=base[qid].get("status", "error"),
            )["anls"],
            "adapted_anls": score_example(
                adapted[qid].get("prediction"),
                answers[qid],
                status=adapted[qid].get("status", "error"),
            )["anls"],
            "visual_cause_claim": (
                "No visual-cause claim without inspecting the page image; "
                "paths are provided for review."
            ),
        }

    Path(args.gallery_md).parent.mkdir(parents=True, exist_ok=True)
    md = [
        "# Holdout gallery (deterministic; illustrative)",
        "",
        "Not representative performance. Selection rules in `holdout_final_analysis.json`.",
        "Inspect linked images before asserting visual causes.",
        "",
    ]
    for name, ent in gallery["categories"].items():
        md.append(f"## {name}")
        if ent is None:
            md.append("")
            md.append("No example available in this category.")
            md.append("")
            continue
        md.extend(
            [
                "",
                f"- qid: {ent['question_id']}",
                f"- Question: {ent['question']}",
                f"- References: {ent['references']!r}",
                f"- Base pred: {ent['base_prediction']!r} (ANLS {ent['base_anls']:.4f})",
                f"- Adapted pred: {ent['adapted_prediction']!r} (ANLS {ent['adapted_anls']:.4f})",
                f"- Image: `{ent['image_relpath']}`",
                f"- Note: {ent['visual_cause_claim']}",
                "",
            ]
        )
    Path(args.gallery_md).write_text("\n".join(md) + "\n")

    # Compact bar chart of primary ANLS (explicit path; create parents)
    figure_path = Path(args.figure_svg)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    write_grouped_bar_svg(
        figure_path,
        categories=["256M", "500M", "InternVL", "LoRA v2 u200"],
        series={
            "primary_ANLS": [
                rows[0]["primary_anls_strict_v2"],
                rows[1]["primary_anls_strict_v2"],
                rows[2]["primary_anls_strict_v2"],
                rows[3]["primary_anls_strict_v2"],
            ]
        },
        title="Holdout primary ANLS strict v2 (n=504)",
        ylabel="ANLS",
    )

    paired = {
        "n_improved": len(improved),
        "n_regressed": len(regressed),
        "n_tied": tied,
        "mean_delta_anls": sum(deltas) / len(deltas),
        "bootstrap": boot,
        "bootstrap_attached": boot_attach["bootstrap_attached"],
        "bootstrap_path": boot_attach.get("bootstrap_path"),
        "bootstrap_omit_reason": boot_attach.get("bootstrap_omit_reason"),
        "bootstrap_correspondence": boot_attach.get("bootstrap_correspondence"),
        "top_improvements": sorted(
            improved, key=lambda r: (-r["delta_anls"], r["question_id"])
        )[:5],
        "top_regressions": sorted(
            regressed, key=lambda r: (r["delta_anls"], r["question_id"])
        )[:5],
    }
    analysis = {
        "task": "holdout_final_analysis",
        "n_requested": len(req),
        "deployment_candidate": "internvl3_1b_step0",
        "adaptation_comparison": "internvl3_1b_lora_v2_u200",
        "systems": rows,
        "paired_internvl_base_vs_lora_v2_u200": paired,
        "gallery": gallery,
        "outputs": {
            "analysis_json": args.out,
            "table_json": args.table_json,
            "gallery_md": args.gallery_md,
            "figure_svg": str(figure_path),
        },
        "caveats": [
            "Visual-token budgets are not matched across model families.",
            "Bootstrap interval is design-based uncertainty, not all uncertainty.",
            "Development-selected model remains InternVL step 0 regardless of holdout ranking.",
            "Diagnostic terminal-period EM is secondary and not official accuracy.",
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.table_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.gallery_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n")
    Path(args.table_json).write_text(
        json.dumps({"task": "holdout_final_table", "rows": rows}, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "out": args.out,
                "figure_svg": str(figure_path),
                "bootstrap_attached": paired["bootstrap_attached"],
                "bootstrap_omit_reason": paired["bootstrap_omit_reason"],
                "n_systems": len(rows),
                "paired": paired["n_improved"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
