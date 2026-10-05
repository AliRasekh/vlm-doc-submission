"""Synthetic regression tests for holdout analysis output isolation and bootstrap CI gating."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from vlm_doc.svg_plot import write_grouped_bar_svg


def _load_analyze_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_holdout_final.py"
    spec = importlib.util.spec_from_file_location("analyze_holdout_final", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestAnalyzeOutputIsolation(unittest.TestCase):
    def test_figure_svg_writes_only_to_explicit_path(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            sealed = tmp / "docs" / "report" / "figures" / "holdout_primary_anls.svg"
            sealed.parent.mkdir(parents=True)
            sealed.write_text("<svg id='sealed'/>")
            out = tmp / "results" / "repro_holdout_analysis" / "holdout_primary_anls.svg"
            out.parent.mkdir(parents=True)
            write_grouped_bar_svg(
                out,
                categories=["a", "b"],
                series={"primary_ANLS": [0.1, 0.2]},
                title="t",
                ylabel="ANLS",
            )
            self.assertTrue(out.is_file())
            self.assertEqual(sealed.read_text(), "<svg id='sealed'/>")
            self.assertNotEqual(out.read_text(), sealed.read_text())


class TestBootstrapCorrespondence(unittest.TestCase):
    def setUp(self):
        self.mod = _load_analyze_module()

    def _manifest(self, n=2, n_groups=2):
        examples = []
        for i in range(n):
            examples.append(
                {
                    "question_id": i + 1,
                    "ucsf_document_id": f"G{(i % n_groups) + 1}",
                }
            )
        return examples

    def _preds(self, scores: dict[int, float]) -> dict[int, dict]:
        return {
            q: {
                "question_id": q,
                "status": "ok",
                "metrics": {"anls_normalized_strict_v2": s, "anls": s},
            }
            for q, s in scores.items()
        }

    def test_omits_when_bootstrap_not_provided(self):
        examples = self._manifest()
        base = self._preds({1: 1.0, 2: 0.0})
        adapted = self._preds({1: 1.0, 2: 1.0})
        out = self.mod.attach_bootstrap_ci(
            bootstrap_path=None,
            bootstrap_obj=None,
            manifest_examples=examples,
            base_preds=base,
            adapted_preds=adapted,
            recomputed_base_mean=0.5,
            recomputed_adapted_mean=1.0,
        )
        self.assertFalse(out["bootstrap_attached"])
        self.assertEqual(out["bootstrap"], {})
        self.assertIn("No --bootstrap-json", out["bootstrap_omit_reason"])

    def test_rejects_unrelated_bootstrap_despite_rounded_score_match(self):
        examples = self._manifest(n=2, n_groups=2)
        base = self._preds({1: 0.7927, 2: 0.7927})
        adapted = self._preds({1: 0.7993, 2: 0.7993})
        # Rounded display values match a plausible sealed CI, but exact floats / groups differ.
        unrelated = {
            "task": "holdout_ucsf_cluster_bootstrap_anls_diff",
            "primary_metric": "anls_normalized_strict_v2",
            "n_questions": 504,  # wrong vs this 2-qid analysis
            "n_groups": 109,
            "point_mean_anls_base": 0.7927,  # rounded-looking value, not exact mean
            "point_mean_anls_adapted": 0.7993,
            "point_diff_adapted_minus_base": 0.0066,
            "bootstrap_diff_p2_5": -0.01,
            "bootstrap_diff_p97_5": 0.02,
        }
        base_mean = self.mod.mean_requested_anls(base, [1, 2])
        adapted_mean = self.mod.mean_requested_anls(adapted, [1, 2])
        out = self.mod.attach_bootstrap_ci(
            bootstrap_path="results/fake_bootstrap.json",
            bootstrap_obj=unrelated,
            manifest_examples=examples,
            base_preds=base,
            adapted_preds=adapted,
            recomputed_base_mean=base_mean,
            recomputed_adapted_mean=adapted_mean,
        )
        self.assertFalse(out["bootstrap_attached"])
        self.assertEqual(out["bootstrap"], {})
        self.assertIn("n_questions", out["bootstrap_omit_reason"])
        self.assertIn("mismatch", out["bootstrap_omit_reason"])

    def test_attaches_when_exact_provenance_matches(self):
        examples = self._manifest(n=2, n_groups=1)
        base = self._preds({1: 0.5, 2: 1.0})
        adapted = self._preds({1: 0.75, 2: 1.0})
        base_mean = self.mod.mean_requested_anls(base, [1, 2])
        adapted_mean = self.mod.mean_requested_anls(adapted, [1, 2])
        boot = {
            "task": "holdout_ucsf_cluster_bootstrap_anls_diff",
            "primary_metric": "anls_normalized_strict_v2",
            "n_questions": 2,
            "n_groups": 1,
            "point_mean_anls_base": base_mean,
            "point_mean_anls_adapted": adapted_mean,
            "point_diff_adapted_minus_base": adapted_mean - base_mean,
            "bootstrap_diff_p2_5": -0.1,
            "bootstrap_diff_p97_5": 0.2,
        }
        out = self.mod.attach_bootstrap_ci(
            bootstrap_path="results/ok_bootstrap.json",
            bootstrap_obj=boot,
            manifest_examples=examples,
            base_preds=base,
            adapted_preds=adapted,
            recomputed_base_mean=base_mean,
            recomputed_adapted_mean=adapted_mean,
        )
        self.assertTrue(out["bootstrap_attached"])
        self.assertEqual(out["bootstrap"]["bootstrap_diff_p2_5"], -0.1)


class TestGalleryPresentationHelpers(unittest.TestCase):
    def setUp(self):
        import sys

        repo = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo / "companion" / "pascal"))

    def test_white_note_and_headings_defined(self):
        from visual_dep import gallery as g

        self.assertIn("Intentionally blank input", g.WHITE_NOTE)
        self.assertEqual(g.CONDITION_HEADINGS["clean"], "A. Correct document")
        self.assertEqual(g.CONDITION_HEADINGS["white"], "B. Blank white input")
        self.assertEqual(g.CONDITION_HEADINGS["unrelated"], "C. Unrelated document")

    def test_render_isolates_presentation_from_source_bytes(self):
        from visual_dep.gallery import render_gallery_figure
        from PIL import Image

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = tmp / "source.png"
            Image.new("RGB", (40, 30), (10, 20, 30)).save(src)
            before = src.read_bytes()
            clean = Image.open(src).convert("RGB")
            white = Image.new("RGB", clean.size, (255, 255, 255))
            unrelated = Image.new("RGB", (50, 20), (200, 100, 50))
            out_pdf = tmp / "g.pdf"
            out_png = tmp / "g.png"
            render_gallery_figure(
                qid=1,
                question="What is shown?",
                panels=[
                    ("clean", clean, "answer-a"),
                    ("white", white, "answer-b"),
                    ("unrelated", unrelated, "answer-c"),
                ],
                out_pdf=out_pdf,
                out_png=out_png,
            )
            self.assertTrue(out_pdf.is_file())
            self.assertTrue(out_png.is_file())
            self.assertEqual(src.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
