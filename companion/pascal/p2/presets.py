"""Development-only demo presets (illustrative; not a representative sample)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

# Selection rules (documented):
# - exact_success: P1 gallery unchanged example with control ANLS=1 (qid 292)
# - failure: smallest qid with fresh Pascal max_num=12 ANLS=0 and nonempty prediction
# - reduced_budget_regression: P1 gallery regression at max_num=4 (qid 5170)
# - informative_tile_reduction: P1 gallery substantial tile reduction (qid 15325)

PRESETS: list[dict[str, Any]] = [
    {
        "id": "exact_success",
        "label": "Exact success (dev)",
        "question_id": 292,
        "image_relpath": "data/images/docvqa_dev/qid_292.png",
        "question": "From which source the data is taken in this document?",
        "default_cap": 12,
        "selection_rule": "P1 gallery unchanged; control ANLS=1.0 / EM=1.0",
    },
    {
        "id": "failure",
        "label": "Failure at default budget (dev)",
        "question_id": 5169,
        "image_relpath": "data/images/docvqa_dev/qid_5169.png",
        "question": "What is chemical formula for Magnesium",
        "default_cap": 12,
        "selection_rule": (
            "Smallest qid with Pascal P1 max_num=12 primary ANLS=0 and nonempty prediction"
        ),
    },
    {
        "id": "reduced_budget_regression",
        "label": "Reduced-budget regression (try cap 4)",
        "question_id": 5170,
        "image_relpath": "data/images/docvqa_dev/qid_5170.png",
        "question": "What is the parts per million analysis for Iron ?",
        "default_cap": 4,
        "selection_rule": "P1 gallery most-negative ΔANLS at max_num=4 vs 12",
    },
    {
        "id": "tile_reduction",
        "label": "Large tile-count drop (informative)",
        "question_id": 15325,
        "image_relpath": "data/images/docvqa_dev/qid_15325.png",
        "question": "What is the name of the puja aggarbatti launched by ITC?",
        "default_cap": 12,
        "selection_rule": "P1 gallery largest actual tile-count reduction 12→4",
    },
]


def _answers_by_qid() -> dict[int, list[str]]:
    path = REPO_ROOT / "data/cache/docvqa_dev_v1_answers.json"
    out: dict[int, list[str]] = {}
    if not path.is_file():
        return out
    for item in json.loads(path.read_text()).get("answers", []):
        out[int(item["question_id"])] = list(item.get("answers", []))
    return out


def list_presets() -> list[dict[str, Any]]:
    refs = _answers_by_qid()
    rows = []
    for p in PRESETS:
        row = dict(p)
        img = (REPO_ROOT / p["image_relpath"]).resolve()
        row["image_path"] = str(img)
        row["image_exists"] = img.is_file()
        row["image_sha256"] = sha256_file(img) if img.is_file() else None
        # References are NEVER passed to the model; UI may show after generation only.
        row["references"] = refs.get(int(p["question_id"]), [])
        rows.append(row)
    return rows


def match_preset_references(
    *, image_path: str | Path, question: str
) -> dict[str, Any] | None:
    """Match by image content hash + exact question (path-agnostic for Gradio copies)."""
    q = str(question).strip()
    try:
        digest = sha256_file(image_path)
    except OSError:
        return None
    for p in list_presets():
        if p.get("image_sha256") == digest and str(p.get("question", "")).strip() == q:
            return p
    return None


def get_preset(preset_id: str) -> dict[str, Any]:
    for p in list_presets():
        if p["id"] == preset_id:
            return p
    raise KeyError(f"unknown preset: {preset_id}")


def load_prerecorded_p1(question_id: int, max_num: int = 12) -> dict[str, Any] | None:
    """Optional offline viewer helper — clearly prerecorded, not live."""
    tag = f"max_num_{int(max_num):02d}"
    path = REPO_ROOT / f"companion/pascal/artifacts/p1_runs/p1_dev_{tag}.jsonl"
    if not path.is_file():
        return None
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            if int(rec.get("question_id", -1)) == int(question_id):
                return {
                    "prerecorded": True,
                    "live_inference": False,
                    "source": str(path.as_posix()),
                    "question_id": question_id,
                    "max_num_cap": max_num,
                    "prediction": rec.get("prediction"),
                    "actual_tile_count": rec.get("actual_tile_count"),
                    "primary_anls_strict_v2": rec.get("primary_anls_strict_v2"),
                    "label": "PRERECORDED P1 OUTPUT — not live inference",
                }
    return None
