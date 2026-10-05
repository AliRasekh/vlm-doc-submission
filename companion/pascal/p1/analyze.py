#!/usr/bin/env python3
"""Analyze Pascal P1 visual-budget runs; write compact tables/figures."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "UNKNOWN"


def _load_jsonl(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    if not path.is_file():
        return out
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[int(rec["question_id"])] = rec
    return out


def _svg_scatter(
    points: list[tuple[float, float, str]],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    out: Path,
    width: int = 640,
    height: int = 420,
) -> None:
    if not points:
        out.write_text("<!-- empty -->\n")
        return
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    pad = 60
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if abs(xmax - xmin) < 1e-12:
        xmin -= 0.05
        xmax += 0.05
    if abs(ymax - ymin) < 1e-12:
        ymin -= 0.05
        ymax += 0.05

    def sx(x: float) -> float:
        return pad + (x - xmin) / (xmax - xmin) * (width - 2 * pad)

    def sy(y: float) -> float:
        return height - pad - (y - ymin) / (ymax - ymin) * (height - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect width="100%" height="100%" fill="#fafafa"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-size="16">{title}</text>',
        f'<text x="{width/2}" y="{height-8}" text-anchor="middle" font-size="12">{xlabel}</text>',
        f'<text x="14" y="{height/2}" transform="rotate(-90 14,{height/2})" '
        f'text-anchor="middle" font-size="12">{ylabel}</text>',
        f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#333"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#333"/>',
    ]
    # Numeric axis ticks (include data extrema so values are readable in the PDF).
    xticks = sorted({xmin, xmax, *xs})
    yticks = sorted({ymin, ymax, *ys})
    for x in xticks:
        parts.append(
            f'<line x1="{sx(x):.1f}" y1="{height-pad}" x2="{sx(x):.1f}" '
            f'y2="{height-pad+5}" stroke="#333"/>'
            f'<text x="{sx(x):.1f}" y="{height-pad+18}" text-anchor="middle" '
            f'font-size="10">{x:.2f}</text>'
        )
    for y in yticks:
        parts.append(
            f'<line x1="{pad-5}" y1="{sy(y):.1f}" x2="{pad}" y2="{sy(y):.1f}" stroke="#333"/>'
            f'<text x="{pad-8}" y="{sy(y)+3:.1f}" text-anchor="end" '
            f'font-size="10">{y:.3f}</text>'
        )
    colors = ["#1f4e79", "#c45c26", "#2f6f4e", "#6b3fa0"]
    for i, (x, y, label) in enumerate(points):
        c = colors[i % len(colors)]
        parts.append(
            f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="6" fill="{c}"/>'
            f'<text x="{sx(x)+8:.1f}" y="{sy(y)-8:.1f}" font-size="12" fill="{c}">'
            f"{label}</text>"
        )
    parts.append("</svg>")
    out.write_text("\n".join(parts) + "\n")


def _svg_bars_memory(
    rows: list[dict[str, Any]], out: Path, width: int = 640, height: int = 420
) -> None:
    pad = 60
    n = max(1, len(rows))
    bw = (width - 2 * pad) / n * 0.35
    max_v = max(
        1.0,
        max(
            float(r.get("peak_allocated_bytes_request_max_over_examples") or 0)
            for r in rows
        ),
        max(
            float(r.get("peak_reserved_bytes_request_max_over_examples") or 0)
            for r in rows
        ),
    )

    def sy(v: float) -> float:
        return height - pad - (v / max_v) * (height - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#fafafa"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-size="16">'
        "P1 peak GPU memory by cap</text>",
        f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#333"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#333"/>',
    ]
    for i, r in enumerate(rows):
        cx = pad + (i + 0.5) * (width - 2 * pad) / n
        a = float(r.get("peak_allocated_bytes_request_max_over_examples") or 0)
        b = float(r.get("peak_reserved_bytes_request_max_over_examples") or 0)
        parts.append(
            f'<rect x="{cx-bw:.1f}" y="{sy(a):.1f}" width="{bw:.1f}" '
            f'height="{height-pad-sy(a):.1f}" fill="#1f4e79"/>'
        )
        parts.append(
            f'<rect x="{cx:.1f}" y="{sy(b):.1f}" width="{bw:.1f}" '
            f'height="{height-pad-sy(b):.1f}" fill="#c45c26"/>'
        )
        parts.append(
            f'<text x="{cx:.1f}" y="{height-pad+18}" text-anchor="middle" font-size="12">'
            f"M={r.get('max_num_cap')}</text>"
        )
    parts.append(
        f'<text x="{width-pad}" y="50" text-anchor="end" font-size="11" fill="#1f4e79">'
        "allocated</text>"
    )
    parts.append(
        f'<text x="{width-pad}" y="66" text-anchor="end" font-size="11" fill="#c45c26">'
        "reserved</text>"
    )
    parts.append("</svg>")
    out.write_text("\n".join(parts) + "\n")


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir", default="companion/pascal/artifacts/p1_runs"
    )
    parser.add_argument(
        "--out-dir", default="companion/pascal/results"
    )
    parser.add_argument(
        "--historical-jsonl", default="results/dev_internvl3_1b.jsonl"
    )
    args = parser.parse_args()
    run_dir = REPO_ROOT / args.run_dir
    out_dir = REPO_ROOT / args.out_dir
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    index_path = run_dir / "run_index.json"
    if not index_path.is_file():
        print(f"ERROR: missing {index_path}", file=sys.stderr)
        return 2
    index = json.loads(index_path.read_text())
    conditions = sorted(
        index["conditions"], key=lambda r: int(r["max_num_cap"]), reverse=True
    )
    control = next(c for c in conditions if int(c["max_num_cap"]) == 12)

    # Load per-condition records
    by_cap: dict[int, dict[int, dict]] = {}
    for c in conditions:
        cap = int(c["max_num_cap"])
        jsonl = REPO_ROOT / c["jsonl_path"]
        by_cap[cap] = _load_jsonl(jsonl)

    control_recs = by_cap[int(control["max_num_cap"])]
    qids = sorted(control_recs.keys())

    paired: dict[str, Any] = {}
    for c in conditions:
        cap = int(c["max_num_cap"])
        if cap == int(control["max_num_cap"]):
            continue
        imp = reg = tie = 0
        deltas = []
        for qid in qids:
            a = float(control_recs[qid].get("primary_anls_strict_v2") or 0.0)
            b = float(by_cap[cap][qid].get("primary_anls_strict_v2") or 0.0)
            d = b - a
            deltas.append(d)
            if d > 1e-12:
                imp += 1
            elif d < -1e-12:
                reg += 1
            else:
                tie += 1
        paired[str(cap)] = {
            "relative_to_cap": int(control["max_num_cap"]),
            "n_improved": imp,
            "n_regressed": reg,
            "n_tied": tie,
            "mean_delta_anls": sum(deltas) / len(deltas) if deltas else None,
        }

    # Historical agreement vs fresh control
    hist = _load_jsonl(REPO_ROOT / args.historical_jsonl)
    exact_agree = score_agree = 0
    score_changes = []
    compared = 0
    missing_hist = 0
    for qid in qids:
        if qid not in hist:
            missing_hist += 1
            continue
        compared += 1
        p_new = (control_recs[qid].get("prediction") or "").strip()
        p_old = (hist[qid].get("prediction") or "").strip()
        if p_new == p_old:
            exact_agree += 1
        s_new = float(control_recs[qid].get("primary_anls_strict_v2") or 0.0)
        # historical may store scores under metrics
        metrics = hist[qid].get("metrics") or {}
        s_old = hist[qid].get("primary_anls_strict_v2")
        if s_old is None:
            s_old = metrics.get("anls_normalized_strict_v2", hist[qid].get("anls"))
        s_old = float(s_old or 0.0)
        if abs(s_new - s_old) < 1e-12:
            score_agree += 1
        else:
            score_changes.append(
                {
                    "question_id": qid,
                    "anls_fresh": s_new,
                    "anls_historical": s_old,
                    "delta_fresh_minus_hist": s_new - s_old,
                }
            )

    table = {
        "task": "pascal_p1_visual_budget_table",
        "label": "supplementary_development_analysis_post_final_eval",
        "source_git_revision_analyze": _git_sha(),
        "source_git_revision_run": index.get("source_git_revision"),
        "runtime": index.get("runtime"),
        "rows": conditions,
        "paired_vs_fresh_control_max12": paired,
        "historical_vs_fresh_control": {
            "historical_jsonl": args.historical_jsonl,
            "n_compared": compared,
            "n_missing_historical": missing_hist,
            "exact_response_agreement": exact_agree,
            "exact_response_agreement_rate": (
                exact_agree / compared if compared else None
            ),
            "primary_anls_agreement": score_agree,
            "primary_anls_agreement_rate": (
                score_agree / compared if compared else None
            ),
            "n_score_changes": len(score_changes),
            "mean_abs_anls_change_on_changes": (
                sum(abs(x["delta_fresh_minus_hist"]) for x in score_changes)
                / len(score_changes)
                if score_changes
                else 0.0
            ),
            "score_change_examples_head": score_changes[:20],
            "note": (
                "Hardware/dtype may differ from Neumann (bf16). "
                "Disagreement is diagnostic, not a controlled HW benchmark."
            ),
        },
    }
    (out_dir / "p1_visual_budget_table.json").write_text(
        json.dumps(table, indent=2) + "\n"
    )

    # Gallery selection (deterministic)
    # Need answers for references display
    answers_path = REPO_ROOT / "data/cache/docvqa_dev_v1_answers.json"
    refs_by = {}
    if answers_path.is_file():
        for item in json.loads(answers_path.read_text()).get("answers", []):
            refs_by[int(item["question_id"])] = list(item.get("answers", []))

    cap_reduced = 4 if 4 in by_cap else min(k for k in by_cap if k != 12)

    def anls(cap: int, qid: int) -> float:
        return float(by_cap[cap][qid].get("primary_anls_strict_v2") or 0.0)

    # regression at reduced cap: most negative delta (cap_reduced - control), tie smallest qid
    reg_qid = None
    reg_delta = 0.0
    for qid in qids:
        d = anls(cap_reduced, qid) - anls(12, qid)
        if d < -1e-12 and (reg_qid is None or d < reg_delta - 1e-15 or (
            abs(d - reg_delta) <= 1e-15 and qid < reg_qid
        )):
            reg_qid, reg_delta = qid, d

    # improvement if present
    imp_qid = None
    imp_delta = 0.0
    for qid in qids:
        d = anls(cap_reduced, qid) - anls(12, qid)
        if d > 1e-12 and (imp_qid is None or d > imp_delta + 1e-15 or (
            abs(d - imp_delta) <= 1e-15 and qid < imp_qid
        )):
            imp_qid, imp_delta = qid, d

    # unchanged: smallest qid with delta==0 and same prediction
    unc_qid = None
    for qid in qids:
        d = anls(cap_reduced, qid) - anls(12, qid)
        if abs(d) <= 1e-12:
            p0 = (control_recs[qid].get("prediction") or "").strip()
            p1 = (by_cap[cap_reduced][qid].get("prediction") or "").strip()
            if p0 == p1:
                unc_qid = qid
                break

    # substantial tile reduction: largest (tiles_control - tiles_reduced), tie smallest qid
    tile_qid = None
    tile_red = -1
    for qid in qids:
        t0 = int(control_recs[qid].get("actual_tile_count") or 0)
        t1 = int(by_cap[cap_reduced][qid].get("actual_tile_count") or 0)
        red = t0 - t1
        if red > tile_red or (red == tile_red and tile_qid is not None and qid < tile_qid) or (
            tile_qid is None and red >= 0
        ):
            if red > tile_red or tile_qid is None:
                tile_qid, tile_red = qid, red
            elif red == tile_red and qid < tile_qid:
                tile_qid, tile_red = qid, red

    def pack(qid: int | None, category: str) -> dict[str, Any]:
        if qid is None:
            return {"category": category, "missing": True}
        # Inspect image existence for visual-cause gate
        img = control_recs[qid].get("image_relpath")
        img_ok = bool(img and (REPO_ROOT / img).is_file())
        return {
            "category": category,
            "missing": False,
            "question_id": qid,
            "question": control_recs[qid].get("question"),
            "references": refs_by.get(qid, []),
            "image_relpath": img,
            "image_inspected": img_ok,
            "visual_cause_claim": (
                "No visual-cause claim without qualitative inspection; path provided."
                if img_ok
                else "Image missing; no visual-cause claim."
            ),
            "control_max12": {
                "prediction": control_recs[qid].get("prediction"),
                "anls": anls(12, qid),
                "em": control_recs[qid].get("primary_em"),
                "tiles": control_recs[qid].get("actual_tile_count"),
            },
            "reduced_cap": {
                "max_num": cap_reduced,
                "prediction": by_cap[cap_reduced][qid].get("prediction"),
                "anls": anls(cap_reduced, qid),
                "em": by_cap[cap_reduced][qid].get("primary_em"),
                "tiles": by_cap[cap_reduced][qid].get("actual_tile_count"),
            },
            "delta_anls_reduced_minus_control": anls(cap_reduced, qid) - anls(12, qid),
        }

    gallery = {
        "task": "pascal_p1_gallery",
        "selection_rules": {
            "regression": (
                f"Most negative ΔANLS at max_num={cap_reduced} vs 12; "
                "ties → smallest qid"
            ),
            "improvement": (
                f"Most positive ΔANLS at max_num={cap_reduced} vs 12 if any; "
                "ties → smallest qid"
            ),
            "unchanged": "Smallest qid with ΔANLS=0 and identical prediction text",
            "tile_reduction": (
                "Largest actual tile-count reduction (12 vs reduced); "
                "ties → smallest qid"
            ),
        },
        "examples": [
            pack(reg_qid, "regression_reduced_cap"),
            pack(imp_qid, "improvement_reduced_cap"),
            pack(unc_qid, "unchanged"),
            pack(tile_qid, "substantial_tile_reduction"),
        ],
    }
    (out_dir / "p1_gallery.json").write_text(json.dumps(gallery, indent=2) + "\n")

    # Figures
    pts_lat = [
        (
            float(c["latency_request_median_s"] or 0.0),
            float(c["primary_anls_strict_v2"] or 0.0),
            f"M={c['max_num_cap']}",
        )
        for c in conditions
    ]
    _svg_scatter(
        pts_lat,
        xlabel="Median request latency (s)",
        ylabel="ANLS strict v2",
        title="P1 ANLS vs median request latency",
        out=fig_dir / "p1_anls_vs_request_latency.svg",
    )
    pts_tiles = [
        (
            float(c["tile_count_mean"] or 0.0),
            float(c["primary_anls_strict_v2"] or 0.0),
            f"M={c['max_num_cap']}",
        )
        for c in conditions
    ]
    _svg_scatter(
        pts_tiles,
        xlabel="Mean actual tile count",
        ylabel="ANLS strict v2",
        title="P1 ANLS vs observed tile usage",
        out=fig_dir / "p1_anls_vs_tile_usage.svg",
    )
    _svg_bars_memory(conditions, fig_dir / "p1_memory_summary.svg")

    # Markdown results
    lines = [
        "# Pascal P1 results — visual budget (development only)",
        "",
        "**Label:** supplementary development analysis after original final evaluation.",
        "Does **not** change deployment selection or official holdout claims.",
        "",
        f"- Run revision: `{index.get('source_git_revision')}`",
        f"- Analyze revision: `{_git_sha()}`",
        f"- GPU: `{index.get('runtime', {}).get('gpu_name')}`",
        f"- dtype / attention: `{index.get('runtime', {}).get('dtype')}` / "
        f"`{index.get('runtime', {}).get('attention_backend')}`",
        "",
        "## Compact table",
        "",
        "| cap | n_ok/fail/miss | ANLS | EM | mean tiles | gen med/p95 s | req med/p95 s | peak alloc / reserved |",
        "|----:|---------------:|-----:|---:|-----------:|--------------:|--------------:|----------------------:|",
    ]
    for c in conditions:
        lines.append(
            "| {cap} | {ok}/{fail}/{miss} | {anls:.4f} | {em:.4f} | {tiles:.3f} | "
            "{gmed:.3f}/{gp95:.3f} | {rmed:.3f}/{rp95:.3f} | {a}/{r} |".format(
                cap=c["max_num_cap"],
                ok=c["n_ok"],
                fail=c["n_failed"],
                miss=c["n_missing"],
                anls=float(c["primary_anls_strict_v2"]),
                em=float(c["primary_em"]),
                tiles=float(c["tile_count_mean"] or 0),
                gmed=float(c["latency_generation_only_median_s"] or 0),
                gp95=float(c["latency_generation_only_p95_s"] or 0),
                rmed=float(c["latency_request_median_s"] or 0),
                rp95=float(c["latency_request_p95_s"] or 0),
                a=c["peak_allocated_bytes_request_max_over_examples"],
                r=c["peak_reserved_bytes_request_max_over_examples"],
            )
        )
    lines += [
        "",
        "## Paired vs fresh control (max_num=12)",
        "",
        "```json",
        json.dumps(paired, indent=2),
        "```",
        "",
        "## Historical vs fresh control",
        "",
        "```json",
        json.dumps(table["historical_vs_fresh_control"], indent=2)[:4000],
        "```",
        "",
        "## Figures",
        "",
        "- `figures/p1_anls_vs_request_latency.svg`",
        "- `figures/p1_anls_vs_tile_usage.svg`",
        "- `figures/p1_memory_summary.svg`",
        "",
        "## Limitations",
        "",
        "- Development-only; not holdout.",
        "- Latency/memory are workload observations on one GPU allocation.",
        "- Tiny ANLS deltas are not statistical superiority claims.",
        "- Filesystem page cache can shrink request latency after the first condition.",
        "- Reserved memory includes allocator caching and is not a per-request VRAM floor.",
        "",
    ]
    (out_dir / "P1_RESULTS.md").write_text("\n".join(lines) + "\n")
    # Also copy-facing docs path
    docs = REPO_ROOT / "docs/companion/pascal/P1_RESULTS.md"
    docs.write_text("\n".join(lines) + "\n")
    print("Wrote compact results under", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
