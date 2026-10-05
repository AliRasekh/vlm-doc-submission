#!/usr/bin/env python3
"""Analyze Phase 3B robustness results; fresh-clean vs historical agreement."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_jsonl(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[int(rec["question_id"])] = rec
    return out


def main() -> int:
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        print("ERROR: export PYTHONNOUSERSITE=1 before python", file=sys.stderr)
        return 2
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/robustness_v1")
    ap.add_argument("--out-json", default="results/robustness_v1_analysis.json")
    ap.add_argument("--out-md", default="docs/examples/robustness_v1_gallery.md")
    ap.add_argument("--plot-svg", default="docs/examples/robustness_v1_anls.svg")
    ap.add_argument("--example-dir", default="docs/examples/robustness_v1_images")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = _repo_root()
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))

    from PIL import Image

    from vlm_doc.image_perturbations import apply_perturbation
    from vlm_doc.metrics import SCORER_VERSION_PRIMARY, score_example
    from vlm_doc.svg_plot import write_grouped_bar_svg

    out_dir = Path(args.out_dir)
    models = ("base", "lora100")
    conds = ("clean", "half_detail", "jpeg_q40")
    stem = {
        "base": "internvl3_1b_base",
        "lora100": "internvl3_1b_lora_u100",
    }

    summaries = {}
    preds = {}
    table = []
    for m in models:
        for c in conds:
            s_path = out_dir / f"{stem[m]}__{c}_summary.json"
            j_path = out_dir / f"{stem[m]}__{c}.jsonl"
            s = json.loads(s_path.read_text())
            summaries[(m, c)] = s
            preds[(m, c)] = _load_jsonl(j_path)
            clean_anls = summaries[(m, "clean")]["anls_normalized_strict_v2"]
            table.append(
                {
                    "model": m,
                    "condition": c,
                    "anls_strict_v2": s["anls_normalized_strict_v2"],
                    "em": s["exact_match"],
                    "n_requested": s["n_requested"],
                    "n_ok": s["n_ok"],
                    "n_failed": s["n_failed"],
                    "n_missing": s["n_missing"],
                    "n_generation_cap_hits": s["n_generation_cap_hits"],
                    "delta_anls_vs_own_clean": s["anls_normalized_strict_v2"] - clean_anls,
                }
            )

    # Fresh clean vs historical full-dev predictions (subset QIDs only)
    hist_base = _load_jsonl(Path("results/dev_internvl3_1b.jsonl"))
    hist_u100 = _load_jsonl(Path("results/dev_internvl3_1b_lora_u100.jsonl"))
    hist = {"base": hist_base, "lora100": hist_u100}
    agreement = {}
    for m in models:
        fresh = preds[(m, "clean")]
        old = hist[m]
        qids = sorted(fresh.keys())
        n_match = 0
        diffs = []
        for q in qids:
            fp = fresh[q].get("prediction")
            hp = (old.get(q) or {}).get("prediction")
            if fp == hp:
                n_match += 1
            else:
                diffs.append(
                    {
                        "question_id": q,
                        "fresh": fp,
                        "historical": hp,
                        "fresh_anls": (fresh[q].get("metrics") or {}).get("anls"),
                        "hist_status": (old.get(q) or {}).get("status"),
                    }
                )
        agreement[m] = {
            "n_compared": len(qids),
            "n_exact_prediction_match": n_match,
            "n_differ": len(diffs),
            "match_rate": (n_match / len(qids)) if qids else None,
            "differences": diffs[:20],
            "n_differences_total": len(diffs),
        }

    # Example selection: deterministic from seed among categories
    rng = random.Random(int(args.seed))
    answers = {
        int(a["question_id"]): a["answers"]
        for a in json.loads(Path("data/cache/docvqa_dev_v1_answers.json").read_text())[
            "answers"
        ]
    }
    # Prefer base model for gallery; look for regression (clean better than half or jpeg)
    base_clean = preds[("base", "clean")]
    base_half = preds[("base", "half_detail")]
    base_jpeg = preds[("base", "jpeg_q40")]

    def anls(rec):
        return float((rec.get("metrics") or {}).get("anls") or 0.0)

    regressions = []
    unchanged = []
    for q in sorted(base_clean):
        c, h, j = base_clean[q], base_half[q], base_jpeg[q]
        if anls(h) < anls(c) - 1e-12 or anls(j) < anls(c) - 1e-12:
            regressions.append(q)
        if (
            c.get("prediction") == h.get("prediction") == j.get("prediction")
            and abs(anls(c) - anls(h)) < 1e-12
            and abs(anls(c) - anls(j)) < 1e-12
        ):
            unchanged.append(q)

    selected = []
    categories_report = {}
    if regressions:
        q = sorted(regressions)[0]
        selected.append(("regression", q))
        categories_report["regression"] = q
    else:
        categories_report["regression"] = None
    if unchanged:
        q = sorted(unchanged)[rng.randrange(len(unchanged))] if unchanged else None
        # deterministic: first after shuffle
        u = list(unchanged)
        rng.shuffle(u)
        q = u[0]
        selected.append(("unchanged", q))
        categories_report["unchanged"] = q
    else:
        categories_report["unchanged"] = None

    # Fill up to 4 with additional deterministic QIDs (sorted), preferring jpeg/half pred change
    changed_pred = [
        q
        for q in sorted(base_clean)
        if base_clean[q].get("prediction") != base_jpeg[q].get("prediction")
        or base_clean[q].get("prediction") != base_half[q].get("prediction")
    ]
    for q in changed_pred:
        if len(selected) >= 4:
            break
        if q not in {x[1] for x in selected}:
            selected.append(("prediction_change", q))
    for q in sorted(base_clean):
        if len(selected) >= 4:
            break
        if q not in {x[1] for x in selected}:
            selected.append(("fill", q))

    example_dir = Path(args.example_dir)
    if example_dir.exists():
        shutil.rmtree(example_dir)
    example_dir.mkdir(parents=True, exist_ok=True)

    gallery = []
    for cat, qid in selected:
        rec = base_clean[qid]
        img_path = Path(rec["image_relpath"])
        clean = Image.open(img_path).convert("RGB")
        paths = {}
        for cond in conds:
            img_t, meta = apply_perturbation(clean, perturbation=cond)
            # Compact JPEG thumb for docs (not used as model input)
            thumb = img_t.copy()
            thumb.thumbnail((512, 512))
            out_p = example_dir / f"qid_{qid}__{cond}.jpg"
            thumb.save(out_p, quality=85)
            paths[cond] = str(out_p.as_posix())
        entry = {
            "category": cat,
            "question_id": qid,
            "question": rec.get("question"),
            "references": answers.get(qid),
            "predictions": {
                cond: preds[("base", cond)][qid].get("prediction") for cond in conds
            },
            "anls": {cond: anls(preds[("base", cond)][qid]) for cond in conds},
            "lora100_predictions": {
                cond: preds[("lora100", cond)][qid].get("prediction") for cond in conds
            },
            "image_thumbs": paths,
        }
        gallery.append(entry)

    write_grouped_bar_svg(
        Path(args.plot_svg),
        categories=["clean", "half_detail", "jpeg_q40"],
        series={
            "base": [
                summaries[("base", c)]["anls_normalized_strict_v2"] for c in conds
            ],
            "lora100": [
                summaries[("lora100", c)]["anls_normalized_strict_v2"] for c in conds
            ],
        },
        title="Phase 3B robustness subset ANLS (strict v2)",
        ylabel="ANLS",
    )

    md = [
        "# Phase 3B robustness gallery (development subset)",
        "",
        "Limited synthetic perturbations on a fixed 16-UCSF-group development subset. "
        "Not a full document-corruption benchmark. Tiny score gaps are not statistical superiority.",
        "",
        f"Example selection seed={args.seed}; categories: {categories_report}.",
        "",
    ]
    if categories_report.get("regression") is None:
        md.append("Requested category **regression**: no examples available.")
        md.append("")
    if categories_report.get("unchanged") is None:
        md.append("Requested category **unchanged**: no examples available.")
        md.append("")
    for g in gallery:
        md.append(f"## qid {g['question_id']} ({g['category']})")
        md.append("")
        md.append(f"- Question: {g['question']}")
        md.append(f"- References: {g['references']!r}")
        md.append(
            f"- Base preds clean/half/jpeg: {g['predictions']['clean']!r} / "
            f"{g['predictions']['half_detail']!r} / {g['predictions']['jpeg_q40']!r}"
        )
        md.append(
            f"- Base ANLS clean/half/jpeg: "
            f"{g['anls']['clean']:.4f} / {g['anls']['half_detail']:.4f} / {g['anls']['jpeg_q40']:.4f}"
        )
        md.append(
            f"- LoRA100 preds clean/half/jpeg: {g['lora100_predictions']['clean']!r} / "
            f"{g['lora100_predictions']['half_detail']!r} / {g['lora100_predictions']['jpeg_q40']!r}"
        )
        md.append("")
        for cond in conds:
            md.append(f"![{cond}]({g['image_thumbs'][cond].replace('docs/examples/', '')})")
        md.append("")

    Path(args.out_md).write_text("\n".join(md) + "\n")

    report = {
        "task": "robustness_v1_analysis",
        "primary_metric": SCORER_VERSION_PRIMARY,
        "results_table": table,
        "fresh_clean_vs_historical": agreement,
        "example_selection": {
            "seed": args.seed,
            "categories": categories_report,
            "selected": gallery,
            "rule": (
                "Prefer one ANLS regression (clean>perturbed) and one unchanged "
                "prediction/score triple on base model; fill with prediction-change "
                "then sorted QIDs; seed=42 for stochastic picks among ties."
            ),
        },
        "plot_svg": args.plot_svg,
        "gallery_md": args.out_md,
        "caveat": (
            "Paired comparisons on identical questions; do not infer statistical "
            "superiority from tiny differences on this small subset."
        ),
    }
    Path(args.out_json).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "out": args.out_json,
                "agreement": {k: {kk: vv for kk, vv in v.items() if kk != "differences"} for k, v in agreement.items()},
                "n_gallery": len(gallery),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
