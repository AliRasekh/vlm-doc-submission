"""Static path-wiring checks (no model inference)."""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_analyze_holdout_defaults_use_holdout_prefix():
    src = (REPO / "scripts/analyze_holdout_final.py").read_text()
    tree = ast.parse(src)
    defaults = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    continue
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                    continue
            # Parse argparse string defaults via simple search (robust enough here)
            break
    assert "--internvl-jsonl" in src
    assert 'default="results/holdout_internvl3_1b.jsonl"' in src
    assert 'default="results/holdout_smolvlm256m.jsonl"' in src
    assert "artifacts/internvl3_1b_lora_v2_u200" in src


def test_submission_guide_documents_path_alias():
    guide = (REPO / "docs/report/SUBMISSION_GUIDE.md").read_text()
    assert "holdout_internvl3_1b.jsonl" in guide
    assert "eval_internvl3_1b_holdout.jsonl" in guide
    assert "analyze_holdout_final.py" in guide


def test_train_script_requires_overwrite_flag():
    src = (REPO / "scripts/train_internvl_lora.py").read_text()
    assert "overwrite_existing_outputs" in src
    assert "refusing to overwrite existing training outputs" in src
    # Guard must run before CUDA / model import.
    assert src.index("Fail safely before model load") < src.index("CUDA required for training")
    assert src.index("Fail safely before model load") < src.index("load_internvl")


def test_submission_guide_lists_all_four_holdout_systems():
    guide = (REPO / "docs/report/SUBMISSION_GUIDE.md").read_text()
    assert "holdout_smolvlm256m.jsonl" in guide
    assert "holdout_smolvlm500m.jsonl" in guide
    assert "holdout_internvl3_1b.jsonl" in guide
    assert "holdout_internvl3_1b_lora_v2_u200.jsonl" in guide
    assert "results/repro_holdout_analysis/" in guide
    assert "--figure-svg results/repro_holdout_analysis/holdout_primary_anls.svg" in guide
    assert "--bootstrap-json" in guide
    assert "Display sealed compact results" in guide
    assert "Fresh model evaluation" in guide


def test_analyze_holdout_isolates_figure_and_bootstrap():
    src = (REPO / "scripts/analyze_holdout_final.py").read_text()
    assert "--figure-svg" in src
    assert 'default="results/holdout_primary_anls_recompute.svg"' in src
    assert "docs/report/figures/holdout_primary_anls.svg" not in src.split("write_grouped_bar_svg")[1][:400]
    assert "--bootstrap-json" in src
    assert "attach_bootstrap_ci" in src
    # Must not auto-load the sealed bootstrap path unconditionally.
    assert 'boot_path = Path("results/holdout_internvl_anls_bootstrap.json")' not in src


def test_compact_holdout_table_present():
    table = REPO / "results/holdout_final_table.json"
    assert table.is_file()
