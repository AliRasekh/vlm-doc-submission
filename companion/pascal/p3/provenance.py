"""Deterministic presentation-order digests for P3 seeds (CPU)."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from . import GRAD_ACCUM, MAX_UPDATES, PRESENTATIONS_PER_RUN
from .paths import REPO_ROOT, seed_dirs


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def compute_order_digest(seed: int, *, n_presentations: int = PRESENTATIONS_PER_RUN) -> dict[str, Any]:
    man = json.loads(
        (REPO_ROOT / "data/manifests/docvqa_train_subset_v1.json").read_text()
    )
    examples = list(man["examples"])
    n = len(examples)
    order = list(range(n))
    rng = random.Random(int(seed))
    rng.shuffle(order)
    qids = [int(examples[order[i % n]]["question_id"]) for i in range(n_presentations)]
    unique = sorted(set(qids))
    return {
        "seed": int(seed),
        "pool_size": n,
        "n_presentations": n_presentations,
        "max_updates": MAX_UPDATES,
        "grad_accum": GRAD_ACCUM,
        "unique_qids": len(unique),
        "presentation_qid_sha256": sha256_text(",".join(str(q) for q in qids)),
        "unique_qid_set_sha256": sha256_text(",".join(str(q) for q in unique)),
        "order_index_sha256": sha256_text(",".join(str(i) for i in order)),
        "first_20_presentation_qids": qids[:20],
        "last_20_presentation_qids": qids[-20:],
        "note": (
            "Replays scripts/train_internvl_lora.py shuffle: "
            "rng=Random(seed); shuffle indices; cursor walks presentations "
            "with wrap. Studies overall seed variability including membership/order."
        ),
    }


def write_order_digest(seed: int) -> Path:
    d = seed_dirs(seed)
    d["root"].mkdir(parents=True, exist_ok=True)
    doc = compute_order_digest(seed)
    d["order_digest"].write_text(json.dumps(doc, indent=2) + "\n")
    return d["order_digest"]
