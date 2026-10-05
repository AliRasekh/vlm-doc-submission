"""CPU tests for P3 seeds, path safety, and order digests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "companion" / "pascal"))

from p3 import PRESENTATIONS_PER_RUN, SEEDS
from p3.paths import assert_under_p3_artifacts
from p3.provenance import compute_order_digest


def test_seeds_are_s_s1_s2():
    assert SEEDS == (42, 43, 44)
    assert PRESENTATIONS_PER_RUN == 1600


def test_order_digests_differ_by_seed():
    d42 = compute_order_digest(42)
    d43 = compute_order_digest(43)
    assert d42["n_presentations"] == 1600
    assert d42["unique_qids"] <= 2000
    assert d42["presentation_qid_sha256"] != d43["presentation_qid_sha256"]
    assert d42["order_index_sha256"] != d43["order_index_sha256"]


def test_path_safety_rejects_original_checkpoints():
    with pytest.raises(RuntimeError):
        assert_under_p3_artifacts("checkpoints/internvl3_1b_lora_v2", kind="ckpt")
    ok = assert_under_p3_artifacts(
        "companion/pascal/artifacts/p3/seed_42/train.jsonl", kind="log"
    )
    assert "artifacts/p3" in str(ok)
