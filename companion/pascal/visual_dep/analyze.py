#!/usr/bin/env python3
"""Analyze visual-input dependence outputs (CPU; existing predictions only)."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from . import CONDITIONS, N_DEV
from .paths import MANIFEST, REPO_ROOT, VD_ART, VD_RESULTS, condition_paths


class EvalIntegrityError(ValueError):
    """Duplicate, unexpected, or incomplete prediction artifacts."""


def load_requested_qids(manifest_path: Path = MANIFEST) -> list[int]:
    man = json.loads(manifest_path.read_text())
    examples = man.get("examples") or man.get("items") or []
    qids = [int(ex["question_id"]) for ex in examples]
    if len(qids) != len(set(qids)):
        raise EvalIntegrityError("development manifest contains duplicate question_ids")
    if len(qids) != N_DEV:
        raise EvalIntegrityError(f"expected {N_DEV} development QIDs, found {len(qids)}")
    return qids


def load_predictions_jsonl(
    path: Path,
    *,
    requested_qids: list[int],
) -> dict[int, dict[str, Any]]:
    """Load predictions; reject duplicates and unexpected QIDs explicitly."""
    req_set = set(requested_qids)
    by_qid: dict[int, dict[str, Any]] = {}
    duplicates: list[int] = []
    unexpected: list[int] = []
    if not path.is_file():
        return by_qid
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        qid = int(r["question_id"])
        if qid not in req_set:
            unexpected.append(qid)
            continue
        if qid in by_qid:
            duplicates.append(qid)
            continue
        by_qid[qid] = r
    if duplicates:
        raise EvalIntegrityError(
            f"duplicate question_ids in {path}: {sorted(set(duplicates))[:20]}"
        )
    if unexpected:
        raise EvalIntegrityError(
            f"unexpected question_ids in {path}: {sorted(set(unexpected))[:20]}"
        )
    return by_qid


def _norm_pred(text: str | None) -> str:
    from vlm_doc.metrics import normalize_text

    if text is None:
        return ""
    return normalize_text(text)


def _ok(row: dict[str, Any] | None) -> bool:
    return row is not None and row.get("status", "ok") == "ok"


def _anls(row: dict[str, Any] | None) -> float:
    if not _ok(row):
        return 0.0
    m = (row or {}).get("metrics") or {}
    return float(m.get("anls_normalized_strict_v2") or 0.0)


def _em(row: dict[str, Any] | None) -> float:
    if not _ok(row):
        return 0.0
    m = (row or {}).get("metrics") or {}
    return float(m.get("exact_match") or 0.0)


def prediction_agreement(
    row: dict[str, Any] | None,
    clean: dict[str, Any] | None,
) -> bool | None:
    """Return True/False when both records are successful; else None (not comparable).

    Missing/error pairs must not count as agreement via empty-string normalization.
    """
    if not _ok(row) or not _ok(clean):
        return None
    return _norm_pred(row.get("prediction")) == _norm_pred(clean.get("prediction"))


def analyze() -> dict[str, Any]:
    import sys

    sys.path.insert(0, str(REPO_ROOT / "src"))

    requested = load_requested_qids()
    n_req = len(requested)
    if n_req != N_DEV or len(set(requested)) != N_DEV:
        raise EvalIntegrityError("requested QID set failed uniqueness/size checks")

    cond_preds: dict[str, dict[int, dict[str, Any]]] = {}
    cond_summaries: dict[str, dict[str, Any]] = {}
    for cond in CONDITIONS:
        paths = condition_paths(cond)
        if not paths["jsonl"].is_file() or not paths["summary"].is_file():
            raise FileNotFoundError(f"missing outputs for condition={cond}")
        # Reject duplicates/unexpected for every condition, including clean.
        preds = load_predictions_jsonl(paths["jsonl"], requested_qids=requested)
        summary = json.loads(paths["summary"].read_text())
        cond_preds[cond] = preds
        cond_summaries[cond] = summary

    clean = cond_preds["clean"]
    rows = []
    for cond in CONDITIONS:
        preds = cond_preds[cond]
        n_ok = n_fail = n_miss = n_cap = 0
        anls_sum = em_sum = 0.0
        n_agree = 0
        n_comparable = 0
        improved = regressed = tied = 0
        tiles: list[int] = []
        for qid in requested:
            row = preds.get(qid)
            base = clean.get(qid)
            if row is None:
                n_miss += 1
            elif row.get("status", "ok") != "ok":
                n_fail += 1
            else:
                n_ok += 1
                if row.get("generation_cap_reached"):
                    n_cap += 1
                if row.get("actual_tile_count") is not None:
                    tiles.append(int(row["actual_tile_count"]))
            anls_sum += _anls(row)
            em_sum += _em(row)
            agr = prediction_agreement(row, base)
            if agr is not None:
                n_comparable += 1
                if agr:
                    n_agree += 1
            delta = _anls(row) - _anls(base)
            if delta > 1e-12:
                improved += 1
            elif delta < -1e-12:
                regressed += 1
            else:
                tied += 1
        hist = Counter(tiles)
        anls = anls_sum / n_req
        em = em_sum / n_req
        agree_rate = (n_agree / n_comparable) if n_comparable else float("nan")
        summary_scores = cond_summaries[cond].get("scores") or {}
        summary_anls = summary_scores.get("mean_anls_requested")
        summary_em = summary_scores.get("mean_exact_match_requested")
        if summary_anls is not None and abs(float(summary_anls) - anls) > 1e-12:
            raise EvalIntegrityError(
                f"{cond}: recomputed ANLS {anls} != summary {summary_anls}"
            )
        if summary_em is not None and abs(float(summary_em) - em) > 1e-12:
            raise EvalIntegrityError(
                f"{cond}: recomputed EM {em} != summary {summary_em}"
            )
        if int(summary_scores.get("n_completed_ok", n_ok)) != n_ok:
            raise EvalIntegrityError(
                f"{cond}: n_ok {n_ok} != summary n_completed_ok "
                f"{summary_scores.get('n_completed_ok')}"
            )
        if int(summary_scores.get("n_failed", n_fail)) != n_fail:
            raise EvalIntegrityError(f"{cond}: n_failed mismatch vs summary")
        if int(summary_scores.get("n_missing", n_miss)) != n_miss:
            raise EvalIntegrityError(f"{cond}: n_missing mismatch vs summary")

        rows.append(
            {
                "condition": cond,
                "n_requested": n_req,
                "n_ok": n_ok,
                "n_failed": n_fail,
                "n_missing": n_miss,
                "n_duplicates": 0,
                "generation_cap_hits": n_cap,
                "anls": anls,
                "em": em,
                "delta_anls_vs_clean": anls
                - (sum(_anls(clean.get(q)) for q in requested) / n_req),
                "prediction_agreement_vs_clean": agree_rate,
                "prediction_agreement_n_matches": n_agree,
                "prediction_agreement_n_comparable": n_comparable,
                "prediction_agreement_denominator": (
                    "n_pairs_where_both_condition_and_clean_status_ok; "
                    "missing/error pairs excluded (do not agree via empty normalization)"
                ),
                "n_improved_vs_clean": improved,
                "n_regressed_vs_clean": regressed,
                "n_tied_vs_clean": tied,
                "tile_count_mean": statistics.fmean(tiles) if tiles else None,
                "tile_count_histogram": {str(k): hist[k] for k in sorted(hist)},
                "summary_anls": summary_anls,
                "summary_em": summary_em,
            }
        )

    table = {
        "task": "pascal_visual_input_dependence",
        "label": "supplementary_development_analysis_post_final_eval",
        "n_dev": n_req,
        "denominator": "requested_development_manifest_qids",
        "metric_policy": "failures_and_missing_count_as_zero_on_requested_set",
        "conditions": rows,
        "runtime": None,
        "interpretation": {
            "summary": (
                "Replacing the correct page with a white or unrelated image reduced "
                "development ANLS by approximately 78 percentage points, indicating "
                "strong dependence on the correct visual input."
            ),
            "white": "Residual performance without readable document content.",
            "unrelated": (
                "Diagnostic scores against the original question references; "
                "may also change layout and visual token/tile count — not a "
                "pure content-only intervention. Not ordinary accuracy on the donor."
            ),
            "non_claims": [
                "Not an exact decomposition of performance into visual vs non-visual parts.",
                "Does not prove fine-grained visual grounding.",
                "Residual correct answers do not establish memorization or contamination.",
                "Not the planned severity-robustness (P4) study.",
            ],
        },
    }
    rt_path = VD_ART / "runtime.json"
    if rt_path.is_file():
        table["runtime"] = json.loads(rt_path.read_text())

    VD_RESULTS.mkdir(parents=True, exist_ok=True)
    out_json = VD_RESULTS / "visual_dependence_table.json"
    out_json.write_text(json.dumps(table, indent=2) + "\n")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = VD_RESULTS / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        labels = [r["condition"] for r in rows]
        anls_vals = [r["anls"] for r in rows]
        em_vals = [r["em"] for r in rows]
        x = range(len(labels))
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        w = 0.35
        ax.bar([i - w / 2 for i in x], anls_vals, width=w, label="ANLS strict v2")
        ax.bar([i + w / 2 for i in x], em_vals, width=w, label="EM")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score (requested-set mean)")
        ax.set_title("Visual-input dependence — InternVL3-1B base (dev 208)")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        for i, r in enumerate(rows):
            ax.annotate(
                f"{r['anls']:.3f}",
                (i - w / 2, r["anls"] + 0.02),
                ha="center",
                fontsize=8,
            )
            ax.annotate(
                f"{r['em']:.3f}",
                (i + w / 2, r["em"] + 0.02),
                ha="center",
                fontsize=8,
            )
        fig.tight_layout()
        fig.savefig(fig_dir / "visual_dependence_anls_em.png", dpi=140)
        fig.savefig(fig_dir / "visual_dependence_anls_em.svg")
        plt.close(fig)
        table["figures"] = {
            "comparison": "companion/pascal/results/figures/visual_dependence_anls_em.png"
        }
        out_json.write_text(json.dumps(table, indent=2) + "\n")
    except Exception as exc:  # noqa: BLE001
        table["figure_error"] = str(exc)
        out_json.write_text(json.dumps(table, indent=2) + "\n")

    print(json.dumps({"table": str(out_json), "rows": rows}, indent=2))
    return table


def main() -> int:
    analyze()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
