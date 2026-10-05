"""Phase 2B metric and aggregation tests."""

from __future__ import annotations

import unittest

from vlm_doc.metrics import (
    aggregate_scores,
    anls_normalized_inclusive_v1,
    anls_normalized_strict_v2,
    exact_match_single,
    score_example,
    validate_answers_sidecar,
)


class TestStrictV2(unittest.TestCase):
    def test_exact(self):
        self.assertEqual(anls_normalized_strict_v2("hello", ["hello"]), 1.0)

    def test_nl_equal_half_is_zero(self):
        pred = "aaaaaaaaaa"
        ref = "aaaaabbbbb"  # NL=0.5
        self.assertEqual(anls_normalized_strict_v2(pred, [ref]), 0.0)
        # inclusive legacy keeps 0.5
        self.assertAlmostEqual(anls_normalized_inclusive_v1(pred, [ref]), 0.5)

    def test_nl_just_below_half_kept(self):
        # 4/10 edits => NL=0.4 => score 0.6
        pred = "aaaaaaaaaa"
        ref = "aaaaaabbbb"
        self.assertAlmostEqual(anls_normalized_strict_v2(pred, [ref]), 0.6)

    def test_empty_refs_invalid(self):
        m = score_example("x", ["", "  "], status="ok")
        self.assertTrue(m["invalid_references"])
        self.assertEqual(m["anls_normalized_strict_v2"], 0.0)


class TestAnswersValidation(unittest.TestCase):
    def test_duplicates_and_missing(self):
        ans = [
            {"question_id": 1, "answers": ["a"]},
            {"question_id": 1, "answers": ["b"]},
        ]
        v = validate_answers_sidecar(ans, required_question_ids=[1, 2])
        self.assertFalse(v["ok"])
        self.assertEqual(v["duplicate_question_ids"], [1])
        self.assertEqual(v["missing_question_ids"], [2])

    def test_empty_reference_flagged(self):
        ans = [{"question_id": 1, "answers": ["", "  "]}]
        v = validate_answers_sidecar(ans, required_question_ids=[1])
        self.assertFalse(v["ok"])
        self.assertEqual(v["invalid_empty_reference_question_ids"], [1])


class TestAggregateGuards(unittest.TestCase):
    def test_duplicate_rejected(self):
        rows = [
            {"question_id": 1, "status": "ok", "metrics": score_example("a", ["a"])},
            {"question_id": 1, "status": "ok", "metrics": score_example("a", ["a"])},
        ]
        with self.assertRaises(ValueError):
            aggregate_scores(rows, n_requested=2, required_question_ids=[1, 2])

    def test_duplicate_rejected_without_required_ids(self):
        rows = [
            {"question_id": 1, "status": "ok", "metrics": score_example("a", ["a"])},
            {"question_id": 1, "status": "ok", "metrics": score_example("a", ["a"])},
        ]
        with self.assertRaises(ValueError):
            aggregate_scores(rows, n_requested=2)

    def test_duplicate_required_question_ids_rejected(self):
        rows = [
            {"question_id": 1, "status": "ok", "metrics": score_example("a", ["a"])},
        ]
        with self.assertRaises(ValueError) as ctx:
            aggregate_scores(rows, n_requested=2, required_question_ids=[1, 1])
        self.assertIn("required_question_ids", str(ctx.exception))

    def test_failed_nonzero_metrics_contribute_zero(self):
        rows = [
            {
                "question_id": 1,
                "status": "ok",
                "metrics": {
                    "anls_normalized_strict_v2": 1.0,
                    "exact_match": 1.0,
                    "anls_normalized_inclusive_v1": 1.0,
                    "anls_vlmevalkit": 1.0,
                },
            },
            {
                "question_id": 2,
                "status": "error",
                "metrics": {
                    "anls_normalized_strict_v2": 0.9,
                    "exact_match": 1.0,
                    "anls_normalized_inclusive_v1": 0.9,
                    "anls_vlmevalkit": 0.9,
                },
            },
        ]
        out = aggregate_scores(rows, n_requested=2, required_question_ids=[1, 2])
        self.assertEqual(out["n_completed_ok"], 1)
        self.assertEqual(out["n_failed"], 1)
        self.assertAlmostEqual(out["mean_anls_requested"], 0.5)
        self.assertAlmostEqual(out["mean_exact_match_requested"], 0.5)

    def test_out_of_manifest_rejected(self):
        rows = [
            {"question_id": 99, "status": "ok", "metrics": score_example("a", ["a"])},
        ]
        with self.assertRaises(ValueError):
            aggregate_scores(rows, n_requested=1, required_question_ids=[1])


class TestExactMatch(unittest.TestCase):
    def test_case(self):
        self.assertEqual(exact_match_single("NNK", ["nnk"]), 1.0)


if __name__ == "__main__":
    unittest.main()
