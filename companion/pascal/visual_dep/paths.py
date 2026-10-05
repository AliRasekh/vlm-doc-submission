"""Paths for visual-input dependence artifacts."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
VD_ART = REPO_ROOT / "companion/pascal/artifacts/visual_dep"
VD_RESULTS = REPO_ROOT / "companion/pascal/results"
MANIFEST = REPO_ROOT / "data/manifests/docvqa_dev_v1.json"
ANSWERS = REPO_ROOT / "data/cache/docvqa_dev_v1_answers.json"
MODEL_CONFIG = REPO_ROOT / "configs/models/internvl3_1b.yaml"


def assert_under_vd_artifacts(path: str | Path, *, kind: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    else:
        p = p.resolve()
    art = VD_ART.resolve()
    try:
        p.relative_to(art)
    except ValueError as exc:
        raise RuntimeError(f"Refusing visual_dep {kind} path outside {art}: {p}") from exc
    return p


def condition_paths(condition: str) -> dict[str, Path]:
    root = VD_ART / f"cond_{condition}"
    return {
        "root": root,
        "jsonl": root / "dev_eval.jsonl",
        "summary": root / "dev_eval_summary.json",
    }
