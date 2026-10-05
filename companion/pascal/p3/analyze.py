#!/usr/bin/env python3
"""P3 analysis: loss curves, ANLS plot, compact table (development only)."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from . import SEEDS
from .paths import P3_ART, P3_RESULTS, REPO_ROOT, seed_dirs

DEV_MANIFEST = REPO_ROOT / "data/manifests/docvqa_dev_v1.json"
N_DEV_EXPECTED = 208


class EvalIntegrityError(ValueError):
    """Duplicate or unexpected QIDs in prediction artifacts."""


def load_requested_qids(manifest_path: Path = DEV_MANIFEST) -> list[int]:
    """Authoritative evaluation QID list from the development manifest."""
    man = json.loads(manifest_path.read_text())
    examples = man.get("examples") or man.get("items") or []
    qids = [int(ex["question_id"]) for ex in examples]
    if len(qids) != len(set(qids)):
        raise EvalIntegrityError("development manifest contains duplicate question_ids")
    if len(qids) != N_DEV_EXPECTED:
        raise EvalIntegrityError(
            f"expected {N_DEV_EXPECTED} development QIDs, found {len(qids)}"
        )
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


def _metric_anls(row: dict[str, Any] | None) -> float:
    if row is None:
        return 0.0
    status = row.get("status", "ok")
    if status != "ok":
        return 0.0
    m = row.get("metrics") or row.get("scores") or {}
    v = m.get("anls_normalized_strict_v2")
    if v is None:
        v = row.get("anls_normalized_strict_v2") or row.get("primary_anls_strict_v2")
    return float(v or 0.0)


def _metric_em(row: dict[str, Any] | None) -> float:
    if row is None:
        return 0.0
    status = row.get("status", "ok")
    if status != "ok":
        return 0.0
    m = row.get("metrics") or row.get("scores") or {}
    v = m.get("exact_match")
    if v is None:
        v = row.get("exact_match") or row.get("primary_em")
    return float(v or 0.0)


def score_against_requested(
    preds: dict[int, dict[str, Any]],
    requested_qids: list[int],
) -> dict[str, Any]:
    """Primary metrics over the full requested set; missing/failed count as 0."""
    n_req = len(requested_qids)
    anls_sum = 0.0
    em_sum = 0.0
    n_ok = n_failed = n_missing = n_cap = 0
    for qid in requested_qids:
        row = preds.get(qid)
        if row is None:
            n_missing += 1
            continue
        if row.get("status", "ok") != "ok":
            n_failed += 1
        else:
            n_ok += 1
            if row.get("generation_cap_reached"):
                n_cap += 1
        anls_sum += _metric_anls(row)
        em_sum += _metric_em(row)
    return {
        "n_requested": n_req,
        "n_ok": n_ok,
        "n_failed": n_failed,
        "n_missing": n_missing,
        "generation_cap_hits": n_cap,
        "mean_anls": anls_sum / n_req,
        "mean_em": em_sum / n_req,
    }


def paired_vs_base(
    preds: dict[int, dict[str, Any]],
    base: dict[int, dict[str, Any]],
    requested_qids: list[int],
) -> dict[str, int]:
    improved = regressed = tied = 0
    for qid in requested_qids:
        delta = _metric_anls(preds.get(qid)) - _metric_anls(base.get(qid))
        if delta > 1e-12:
            improved += 1
        elif delta < -1e-12:
            regressed += 1
        else:
            tied += 1
    return {"n_improved": improved, "n_regressed": regressed, "n_tied": tied}


def _summary_counts(summary: dict[str, Any]) -> dict[str, Any]:
    scores = summary.get("scores") or {}
    return {
        "n_ok": scores.get("n_completed_ok", summary.get("n_ok")),
        "n_failed": scores.get("n_failed", summary.get("n_failed")),
        "n_missing": scores.get("n_missing", summary.get("n_missing")),
        "primary_anls": scores.get("mean_anls_requested")
        or summary.get("primary_anls_strict_v2"),
        "primary_em": scores.get("mean_exact_match_requested")
        or summary.get("primary_em"),
    }


def _mean_window(losses: list[float], start: int, n: int = 20) -> float | None:
    chunk = losses[start : start + n]
    if len(chunk) < n:
        return None
    return sum(chunk) / len(chunk)


def trailing_mean(losses: list[float], window: int = 20) -> list[float | None]:
    """Trailing mean ending at each update index (1-indexed series length)."""
    out: list[float | None] = []
    for i in range(len(losses)):
        if i + 1 < window:
            out.append(None)
        else:
            chunk = losses[i + 1 - window : i + 1]
            out.append(sum(chunk) / window)
    return out


def _adapter_hash(seed: int) -> str | None:
    d = seed_dirs(seed)["checkpoint_root"] / "update_0200"
    meta = d / "vlm_doc_checkpoint_meta.json"
    if meta.is_file():
        m = json.loads(meta.read_text())
        wh = m.get("adapter_weight_sha256") or {}
        if "adapter_model.safetensors" in wh:
            return wh["adapter_model.safetensors"]
        if wh:
            return next(iter(wh.values()))
    for p in sorted(d.glob("*.safetensors")):
        import hashlib

        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    return None


def analyze() -> dict[str, Any]:
    P3_RESULTS.mkdir(parents=True, exist_ok=True)
    requested = load_requested_qids()
    base_sum = P3_ART / "eval_fresh_base/dev_eval_summary.json"
    base_jsonl = P3_ART / "eval_fresh_base/dev_eval.jsonl"
    base_rows = load_predictions_jsonl(base_jsonl, requested_qids=requested)
    base_summary = json.loads(base_sum.read_text()) if base_sum.is_file() else {}
    base_counts = _summary_counts(base_summary)
    base_scored = score_against_requested(base_rows, requested)

    # Prefer recomputed requested-set scores; fall back to summary means if empty.
    fresh_anls = base_scored["mean_anls"] if base_rows or base_scored["n_missing"] else base_counts["primary_anls"]
    fresh_em = base_scored["mean_em"] if base_rows or base_scored["n_missing"] else base_counts["primary_em"]

    p1_path = REPO_ROOT / "companion/pascal/results/p1_visual_budget_table.json"
    p1_diag = None
    if p1_path.is_file():
        p1 = json.loads(p1_path.read_text())
        for row in p1.get("rows") or []:
            if int(row.get("max_num_cap", -1)) == 12:
                p1_diag = {
                    "p1_cap12_anls": row.get("primary_anls_strict_v2"),
                    "p1_cap12_em": row.get("primary_em"),
                    "fresh_base_anls": fresh_anls,
                }
                break

    neumann_ref = None
    nev = REPO_ROOT / "results/dev_internvl_lora_v2_comparison.json"
    if nev.is_file():
        n = json.loads(nev.read_text())
        neumann_ref = {
            "label": "historical_neumann_v2_u200_separate_reference",
            "excluded_from_mean_sd": True,
            "source": str(nev.as_posix()),
            "scores": n.get("scores") or n.get("selected") or n,
        }

    seed_rows = []
    loss_curves: dict[str, Any] = {}
    for seed in SEEDS:
        d = seed_dirs(seed)
        losses: list[float] = []
        lrs: list[float] = []
        if d["log_jsonl"].is_file():
            for line in d["log_jsonl"].read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                losses.append(float(rec["loss"]))
                lrs.append(float(rec["lr"]))
        train_sum = (
            json.loads(d["summary_json"].read_text())
            if d["summary_json"].is_file()
            else {}
        )
        order = (
            json.loads(d["order_digest"].read_text())
            if d["order_digest"].is_file()
            else {}
        )
        preds = load_predictions_jsonl(d["eval_jsonl"], requested_qids=requested)
        eval_sum = (
            json.loads(d["eval_summary"].read_text())
            if d["eval_summary"].is_file()
            else {}
        )
        scored = score_against_requested(preds, requested)
        paired = paired_vs_base(preds, base_rows, requested)
        anls = scored["mean_anls"]
        em = scored["mean_em"]
        base_anls = base_scored["mean_anls"]
        row = {
            "seed": seed,
            "adapter_weight_sha256": _adapter_hash(seed),
            "completed_updates": train_sum.get("completed_updates"),
            "samples_seen": train_sum.get("samples_seen"),
            "unique_qids": train_sum.get("unique_examples_seen")
            or order.get("unique_qids"),
            "presentation_qid_sha256": order.get("presentation_qid_sha256"),
            "unique_qid_set_sha256": order.get("unique_qid_set_sha256"),
            "mean_loss_first_20": _mean_window(losses, 0, 20),
            "mean_loss_last_20": _mean_window(losses, max(0, len(losses) - 20), 20)
            if len(losses) >= 20
            else None,
            "dev_anls": anls,
            "dev_em": em,
            "fresh_base_anls": base_anls,
            "delta_anls_adapted_minus_base": anls - base_anls,
            "n_improved": paired["n_improved"],
            "n_regressed": paired["n_regressed"],
            "n_tied": paired["n_tied"],
            "n_failed_or_non_ok": scored["n_failed"],
            "n_missing": scored["n_missing"],
            "generation_cap_hits": scored["generation_cap_hits"],
            "n_scored_requested": scored["n_requested"],
            "n_ok": scored["n_ok"],
            "eval_summary_primary_anls": (eval_sum.get("scores") or {}).get(
                "mean_anls_requested"
            ),
        }
        seed_rows.append(row)
        loss_curves[str(seed)] = {
            "loss": losses,
            "lr": lrs,
            "trailing_mean_20": trailing_mean(losses, 20),
        }

    anls_vals = [r["dev_anls"] for r in seed_rows if not math.isnan(r["dev_anls"])]
    summary = {
        "task": "pascal_p3_seed_reproducibility",
        "label": "supplementary_development_analysis_post_final_eval",
        "seeds": list(SEEDS),
        "n_dev": len(requested),
        "denominator": "requested_development_manifest_qids",
        "metric_policy": "failures_and_missing_count_as_zero_on_requested_set",
        "fresh_base": {
            "summary_path": str(base_sum.as_posix()),
            "primary_anls": fresh_anls,
            "primary_em": fresh_em,
            "n_ok": base_counts["n_ok"]
            if base_counts["n_ok"] is not None
            else base_scored["n_ok"],
            "n_failed": base_counts["n_failed"]
            if base_counts["n_failed"] is not None
            else base_scored["n_failed"],
            "n_missing": base_counts["n_missing"]
            if base_counts["n_missing"] is not None
            else base_scored["n_missing"],
            "recomputed_n_ok": base_scored["n_ok"],
            "recomputed_n_failed": base_scored["n_failed"],
            "recomputed_n_missing": base_scored["n_missing"],
        },
        "p1_cap12_diagnostic": p1_diag,
        "neumann_v2_u200_reference_separate": neumann_ref,
        "per_seed": seed_rows,
        "three_seed_adapted_anls": {
            "values": anls_vals,
            "mean": statistics.fmean(anls_vals) if anls_vals else None,
            "sample_sd_ddof1": statistics.stdev(anls_vals) if len(anls_vals) >= 2 else None,
            "min": min(anls_vals) if anls_vals else None,
            "max": max(anls_vals) if anls_vals else None,
            "note": (
                "Descriptive summary of three Pascal seeds only. "
                "Does not establish broad statistical significance. "
                "Neumann historical result excluded."
            ),
        },
        "loss_semantics": (
            "Per-update mean loss over grad_accum microbatches; different seeds "
            "may present different examples — training statistic, not matched pairs. "
            "Figures show raw loss (low opacity) plus trailing 20-update mean."
        ),
        "training_pool_coverage_note": (
            "1600 unique QIDs per completed run is 80% of the 2000-example "
            "training pool under shuffle+wrap — not full-pool coverage."
        ),
    }

    table_path = P3_RESULTS / "p3_seed_table.json"
    curves_path = P3_RESULTS / "p3_loss_curves.json"
    table_path.write_text(json.dumps(summary, indent=2) + "\n")
    curves_path.write_text(
        json.dumps(
            {
                "task": "pascal_p3_loss_curves",
                "curves_by_seed": loss_curves,
                "loss_semantics": summary["loss_semantics"],
            },
            indent=2,
        )
        + "\n"
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = P3_RESULTS / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        colors = {"42": "C0", "43": "C1", "44": "C2"}
        for seed, curve in loss_curves.items():
            ys = curve.get("loss") or []
            if not ys:
                continue
            xs = list(range(1, len(ys) + 1))
            ax.plot(
                xs,
                ys,
                color=colors.get(str(seed), None),
                alpha=0.25,
                linewidth=1.0,
                label=f"seed {seed} raw",
            )
            tm = curve.get("trailing_mean_20") or trailing_mean(ys, 20)
            xs_m = [x for x, v in zip(xs, tm) if v is not None]
            ys_m = [v for v in tm if v is not None]
            ax.plot(
                xs_m,
                ys_m,
                color=colors.get(str(seed), None),
                alpha=0.95,
                linewidth=2.0,
                label=f"seed {seed} trailing mean (20)",
            )
        ax.set_xlabel("Optimizer update")
        ax.set_ylabel("Train loss (microbatch-mean)")
        ax.set_title(
            "P3 training loss — raw (faint) + trailing 20-update mean\n"
            "(samples / order may differ across seeds)"
        )
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "p3_loss_curves.png", dpi=140)
        fig.savefig(fig_dir / "p3_loss_curves.svg")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        xs = list(range(len(seed_rows)))
        ys = [r["dev_anls"] for r in seed_rows]
        ax.scatter(xs, ys, s=70, label="Pascal adapted (u200)", zorder=3)
        for i, r in enumerate(seed_rows):
            # Extra upward offset for the highest point (seed 44) to avoid axis clip.
            dy = 10 if r["seed"] == 44 else 4
            ax.annotate(
                str(r["seed"]),
                (i, r["dev_anls"]),
                textcoords="offset points",
                xytext=(8, dy),
                fontsize=9,
            )
        base_y = summary["fresh_base"].get("primary_anls")
        if base_y is not None:
            ax.axhline(base_y, color="black", ls="--", label="Fresh Pascal base")
        mean_y = summary["three_seed_adapted_anls"].get("mean")
        if mean_y is not None:
            ax.axhline(mean_y, color="tab:blue", ls=":", label="3-seed mean")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"s={r['seed']}" for r in seed_rows])
        ax.set_ylabel("Dev ANLS (strict v2)")
        ax.set_title("P3 development ANLS (three seeds)")
        y_min = min([y for y in ys if y is not None] + ([base_y] if base_y else []))
        y_max = max([y for y in ys if y is not None] + ([base_y] if base_y else []))
        pad = max(0.01, 0.15 * (y_max - y_min + 1e-9))
        ax.set_ylim(y_min - pad, y_max + pad * 1.8)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "p3_dev_anls.png", dpi=140)
        fig.savefig(fig_dir / "p3_dev_anls.svg")
        plt.close(fig)
        summary["figures"] = {
            "loss_curves": "companion/pascal/results/figures/p3_loss_curves.png",
            "dev_anls": "companion/pascal/results/figures/p3_dev_anls.png",
        }
        table_path.write_text(json.dumps(summary, indent=2) + "\n")
    except Exception as exc:  # noqa: BLE001
        summary["figure_error"] = str(exc)
        table_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(
        json.dumps(
            {
                "table": str(table_path),
                "three_seed": summary["three_seed_adapted_anls"],
                "fresh_base_counts": {
                    k: summary["fresh_base"][k]
                    for k in ("n_ok", "n_failed", "n_missing", "primary_anls", "primary_em")
                },
            },
            indent=2,
        )
    )
    return summary


def main() -> int:
    analyze()
    return 0


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
