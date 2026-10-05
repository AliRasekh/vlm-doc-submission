"""Path safety for P3 — keep destructive writes inside companion artifacts."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
P3_ART = REPO_ROOT / "companion/pascal/artifacts/p3"
P3_RESULTS = REPO_ROOT / "companion/pascal/results"


def assert_under_p3_artifacts(path: str | Path, *, kind: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    else:
        p = p.resolve()
    art = P3_ART.resolve()
    try:
        p.relative_to(art)
    except ValueError as exc:
        raise RuntimeError(f"Refusing P3 {kind} path outside {art}: {p}") from exc
    return p


def seed_dirs(seed: int) -> dict[str, Path]:
    root = P3_ART / f"seed_{int(seed):02d}"
    return {
        "root": root,
        "checkpoint_root": root / "checkpoints",
        "log_jsonl": root / "train.jsonl",
        "summary_json": root / "train_summary.json",
        "supervised_example_md": root / "supervised_example.md",
        "supervised_example_json": root / "supervised_example.json",
        "order_digest": root / "order_digest.json",
        "eval_jsonl": root / "dev_eval.jsonl",
        "eval_summary": root / "dev_eval_summary.json",
    }
