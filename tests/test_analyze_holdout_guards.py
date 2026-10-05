"""Synthetic regression tests for holdout analysis output isolation and bootstrap CI gating."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from vlm_doc.metrics import SCORER_VERSION_PRIMARY, score_example
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

    def _manifest(self, n=2, n_groups=1):
        examples = []
        for i in range(n):
            examples.append(
                {
                    "question_id": i + 1,
                    "ucsf_document_id": f"G{(i % n_groups) + 1}",
                }
            )
        return examples

    def _answers(self, n=2):
        return {i + 1: ["ref"] for i in range(n)}

    def _preds_from_anls(self, scores: dict[int, float]) -> dict[int, dict]:
        """Build ok records whose canonical rescoring yields the requested ANLS.

        Uses prediction=ref for 1.0 and a distant string for 0.0.
        """
        out = {}
        for q, s in scores.items():
            if s == 1.0:
                pred = "ref"
            elif s == 0.0:
                pred = "zzzzzzzzzz"
            else:
                raise ValueError("synthetic helper only supports 0/1 ANLS")
            out[q] = {"question_id": q, "status": "ok", "prediction": pred}
        return out

    def test_omits_when_bootstrap_not_provided(self):
        out = self.mod.attach_bootstrap_ci(
            bootstrap_path=None,
            bootstrap_obj=None,
            current_provenance={
                "schema": "holdout_bootstrap_inputs_v1",
                "primary_metric": SCORER_VERSION_PRIMARY,
                "paired_primary_scores_sha256": "abc",
                "manifest_question_groups_sha256": "def",
            },
        )
        self.assertFalse(out["bootstrap_attached"])
        self.assertEqual(out["bootstrap"], {})
        self.assertIn("No --bootstrap-json", out["bootstrap_omit_reason"])

    def test_omits_historical_bootstrap_without_input_hashes(self):
        """Historical sealed bootstrap has aggregates only — must not attach."""
        examples = self._manifest()
        answers = self._answers()
        base = self._preds_from_anls({1: 0.0, 2: 1.0})
        adapted = self._preds_from_anls({1: 0.0, 2: 1.0})
        prov = self.mod.paired_primary_scores_provenance(
            manifest_examples=examples,
            base_preds=base,
            adapted_preds=adapted,
            answers=answers,
            score_example_fn=score_example,
            primary=SCORER_VERSION_PRIMARY,
        )
        historical = {
            "task": "holdout_ucsf_cluster_bootstrap_anls_diff",
            "primary_metric": "anls_normalized_strict_v2",
            "n_questions": 2,
            "n_groups": 1,
            "point_mean_anls_base": 0.5,
            "point_mean_anls_adapted": 0.5,
            "point_diff_adapted_minus_base": 0.0,
            "bootstrap_diff_p2_5": -0.2,
            "bootstrap_diff_p97_5": 0.2,
        }
        out = self.mod.attach_bootstrap_ci(
            bootstrap_path="results/holdout_internvl_anls_bootstrap.json",
            bootstrap_obj=historical,
            current_provenance=prov,
        )
        self.assertFalse(out["bootstrap_attached"])
        self.assertIn("lacks trustworthy input provenance", out["bootstrap_omit_reason"])

    def test_rejects_old_ci_when_means_match_but_paired_scores_differ(self):
        """base=[0,1], adapted=[0,1] vs base=[0,1], adapted=[1,0]: means match, CI must not transfer."""
        examples = self._manifest(n=2, n_groups=1)
        answers = self._answers()
        old_base = self._preds_from_anls({1: 0.0, 2: 1.0})
        old_adapted = self._preds_from_anls({1: 0.0, 2: 1.0})
        new_base = self._preds_from_anls({1: 0.0, 2: 1.0})
        new_adapted = self._preds_from_anls({1: 1.0, 2: 0.0})

        old_prov = self.mod.paired_primary_scores_provenance(
            manifest_examples=examples,
            base_preds=old_base,
            adapted_preds=old_adapted,
            answers=answers,
            score_example_fn=score_example,
            primary=SCORER_VERSION_PRIMARY,
        )
        new_prov = self.mod.paired_primary_scores_provenance(
            manifest_examples=examples,
            base_preds=new_base,
            adapted_preds=new_adapted,
            answers=answers,
            score_example_fn=score_example,
            primary=SCORER_VERSION_PRIMARY,
        )
        # Means match for both pairings.
        old_base_mean = self.mod.mean_requested_anls(
            old_base,
            [1, 2],
            answers,
            score_example_fn=score_example,
            primary=SCORER_VERSION_PRIMARY,
        )
        new_base_mean = self.mod.mean_requested_anls(
            new_base,
            [1, 2],
            answers,
            score_example_fn=score_example,
            primary=SCORER_VERSION_PRIMARY,
        )
        old_ad_mean = self.mod.mean_requested_anls(
            old_adapted,
            [1, 2],
            answers,
            score_example_fn=score_example,
            primary=SCORER_VERSION_PRIMARY,
        )
        new_ad_mean = self.mod.mean_requested_anls(
            new_adapted,
            [1, 2],
            answers,
            score_example_fn=score_example,
            primary=SCORER_VERSION_PRIMARY,
        )
        self.assertEqual(old_base_mean, new_base_mean)
        self.assertEqual(old_ad_mean, new_ad_mean)
        self.assertNotEqual(
            old_prov["paired_primary_scores_sha256"],
            new_prov["paired_primary_scores_sha256"],
        )

        old_boot = {
            "task": "holdout_ucsf_cluster_bootstrap_anls_diff",
            "primary_metric": SCORER_VERSION_PRIMARY,
            "n_questions": 2,
            "n_groups": 1,
            "point_mean_anls_base": old_base_mean,
            "point_mean_anls_adapted": old_ad_mean,
            "point_diff_adapted_minus_base": 0.0,
            "bootstrap_diff_p2_5": -0.11,
            "bootstrap_diff_p97_5": 0.11,
            "input_provenance": old_prov,
        }
        out = self.mod.attach_bootstrap_ci(
            bootstrap_path="results/old_bootstrap.json",
            bootstrap_obj=old_boot,
            current_provenance=new_prov,
        )
        self.assertFalse(out["bootstrap_attached"])
        self.assertEqual(out["bootstrap"], {})
        self.assertIn("paired_primary_scores_sha256", out["bootstrap_omit_reason"])

    def test_attaches_when_input_hashes_match(self):
        examples = self._manifest()
        answers = self._answers()
        base = self._preds_from_anls({1: 0.0, 2: 1.0})
        adapted = self._preds_from_anls({1: 1.0, 2: 1.0})
        prov = self.mod.paired_primary_scores_provenance(
            manifest_examples=examples,
            base_preds=base,
            adapted_preds=adapted,
            answers=answers,
            score_example_fn=score_example,
            primary=SCORER_VERSION_PRIMARY,
        )
        boot = {
            "task": "holdout_ucsf_cluster_bootstrap_anls_diff",
            "primary_metric": SCORER_VERSION_PRIMARY,
            "bootstrap_diff_p2_5": -0.1,
            "bootstrap_diff_p97_5": 0.2,
            "input_provenance": prov,
        }
        out = self.mod.attach_bootstrap_ci(
            bootstrap_path="results/ok_bootstrap.json",
            bootstrap_obj=boot,
            current_provenance=prov,
        )
        self.assertTrue(out["bootstrap_attached"])
        self.assertEqual(out["bootstrap"]["bootstrap_diff_p2_5"], -0.1)


class TestCanonicalRescoring(unittest.TestCase):
    def setUp(self):
        self.mod = _load_analyze_module()

    def test_failed_record_with_cached_nonzero_metrics_contributes_zero(self):
        answers = {1: ["gold"], 2: ["gold"]}
        preds = {
            1: {
                "question_id": 1,
                "status": "ok",
                "prediction": "gold",
                "metrics": {
                    "anls_normalized_strict_v2": 1.0,
                    "exact_match": 1.0,
                    "anls_vlmevalkit": 1.0,
                },
            },
            2: {
                "question_id": 2,
                "status": "error",
                "prediction": "gold",
                # Stale cached metrics must be ignored.
                "metrics": {
                    "anls_normalized_strict_v2": 0.9,
                    "exact_match": 1.0,
                    "anls_vlmevalkit": 0.9,
                },
            },
        }
        mean = self.mod.mean_requested_anls(
            preds,
            [1, 2],
            answers,
            score_example_fn=score_example,
            primary=SCORER_VERSION_PRIMARY,
        )
        self.assertAlmostEqual(mean, 0.5)
        failed = self.mod.rescore_primary(
            preds[2],
            answers[2],
            score_example_fn=score_example,
            primary=SCORER_VERSION_PRIMARY,
        )
        self.assertEqual(failed["anls_normalized_strict_v2"], 0.0)
        self.assertEqual(failed["exact_match"], 0.0)


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
