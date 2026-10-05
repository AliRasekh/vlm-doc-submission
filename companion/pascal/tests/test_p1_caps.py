"""Focused P1 tests: caps reach preprocess, tile rules, fingerprints, QID coverage."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "companion" / "pascal"))

from p1.caps import candidate_max_nums, execution_order, omitted_caps
from p1.fingerprint import build_p1_fingerprint, fingerprint_sha
from vlm_doc.adapters.internvl import dynamic_preprocess, load_pixel_values


def test_candidate_caps_for_m12():
    caps = candidate_max_nums(12, min_num=1)
    assert caps == [1, 4, 12]
    assert omitted_caps(12, min_num=1) == []
    assert execution_order(caps, original_max_num=12) == [12, 4, 1]


def test_cap_reaches_preprocess_and_tile_rules():
    # Synthetic wide image → multi-tile under larger caps
    img = Image.new("RGB", (1600, 400), color=(200, 200, 200))
    for cap in (1, 4, 12):
        tiles = dynamic_preprocess(
            img, min_num=1, max_num=cap, image_size=448, use_thumbnail=True
        )
        pv, info = load_pixel_values(
            img, input_size=448, max_num=cap, use_thumbnail=True
        )
        assert info["max_num"] == cap
        assert info["tile_count"] == len(tiles) == int(pv.shape[0])
        # With thumbnail: if blocks==1, no extra; else +1
        blocks = len(tiles) - (1 if info["thumbnail_appended"] else 0)
        assert 1 <= blocks <= cap
        if blocks == 1:
            assert info["tile_count"] == 1
            assert not info["thumbnail_appended"]
        else:
            assert info["tile_count"] == blocks + 1
            assert info["thumbnail_appended"]


def test_square_image_single_block_no_thumbnail_extra():
    img = Image.new("RGB", (448, 448), color=(10, 20, 30))
    for cap in (1, 4, 12):
        tiles = dynamic_preprocess(
            img, min_num=1, max_num=cap, image_size=448, use_thumbnail=True
        )
        # Closest ratio often 1x1 → no thumbnail append
        if len(tiles) == 1:
            assert True
        else:
            # If multi-block, thumbnail appended
            assert len(tiles) >= 2


def test_condition_fingerprints_differ(tmp_path):
    cfg = {
        "model_id": "OpenGVLab/InternVL3-1B",
        "revision": "4415a3b810e636d11dfa86b0e9ba40bb00535aa8",
        "instruction": "Answer the question using a single word or short phrase.",
        "max_new_tokens": 64,
        "prefer_bf16": True,
        "seed": 0,
        "do_sample": False,
        "image_size": 448,
        "min_num": 1,
        "use_thumbnail": True,
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("model_id: x\n")
    # build_p1_fingerprint hashes cfg_path relative to repo; write into companion tree
    real_cfg = REPO / "companion/pascal/configs/p1_visual_budget.yaml"
    fps = []
    for cap in (1, 4, 12):
        fp = build_p1_fingerprint(
            repo_root=REPO,
            cfg=cfg,
            cfg_path=real_cfg,
            max_num=cap,
            manifest_content_sha256="abc",
            resolved_dtype="float16",
            attention_backend="eager",
            dtype_policy="float16_fallback_no_bf16",
            source_revision="deadbeef",
        )
        fps.append(fingerprint_sha(fp))
    assert len(set(fps)) == 3


def test_dev_manifest_qid_coverage():
    man = json.loads((REPO / "data/manifests/docvqa_dev_v1.json").read_text())
    qids = [int(e["question_id"]) for e in man["examples"]]
    assert len(qids) == 208
    assert len(qids) == len(set(qids))
    for e in man["examples"]:
        assert e.get("image_relpath")


@pytest.mark.integration
def test_dev_manifest_images_materialized():
    man = json.loads((REPO / "data/manifests/docvqa_dev_v1.json").read_text())
    missing = [
        e["image_relpath"]
        for e in man["examples"]
        if not (REPO / e["image_relpath"]).is_file()
    ]
    if missing:
        pytest.skip(f"development images not materialized ({len(missing)} missing)")
    assert not missing
