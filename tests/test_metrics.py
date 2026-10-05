"""Unit tests for DocVQA metrics (no GPU required)."""

from __future__ import annotations

import unittest

from vlm_doc.metrics import (
    ANLS_THRESHOLD,
    aggregate_scores,
    anls_single,
    anls_vlmevalkit_single,
    exact_match_single,
    normalize_text,
    score_example,
)


class TestNormalize(unittest.TestCase):
    def test_case_whitespace(self):
        self.assertEqual(normalize_text("  Foo   BAR\n"), "foo bar")


class TestExactMatch(unittest.TestCase):
    def test_exact(self):
        self.assertEqual(exact_match_single("NNK", ["NNK"]), 1.0)

    def test_case(self):
        self.assertEqual(exact_match_single("nnk", ["NNK"]), 1.0)

    def test_whitespace(self):
        self.assertEqual(exact_match_single("  nnk  ", ["NNK"]), 1.0)

    def test_punctuation_not_stripped(self):
        # EM does not strip punctuation; trailing period differs.
        self.assertEqual(exact_match_single("83.4 %.", ["83.4%"]), 0.0)

    def test_numeric(self):
        self.assertEqual(exact_match_single("2", ["2"]), 1.0)

    def test_empty_pred(self):
        self.assertEqual(exact_match_single("", ["NNK"]), 0.0)

    def test_multiple_refs(self):
        self.assertEqual(
            exact_match_single("nnk", ["foo", "NNK", "bar"]), 1.0
        )


class TestANLS(unittest.TestCase):
    def test_exact(self):
        self.assertEqual(anls_single("hello", ["hello"]), 1.0)

    def test_case(self):
        self.assertEqual(anls_single("HELLO", ["hello"]), 1.0)

    def test_whitespace(self):
        self.assertEqual(anls_single("hello   world", ["hello world"]), 1.0)

    def test_empty_pred(self):
        self.assertEqual(anls_single("", ["hello"]), 0.0)

    def test_both_empty_refs_invalid(self):
        # Empty/whitespace-only references are invalid and do not score as matches.
        self.assertEqual(anls_single("", [""]), 0.0)

    def test_threshold_boundary_equal_half(self):
        # NL == 0.5 ⇒ similarity == 0.5 ⇒ primary keeps 0.5 (sim >= 0.5).
        pred = "aaaaaaaaaa"
        ref = "aaaaabbbbb"  # 5 substitutions / 10
        self.assertAlmostEqual(anls_single(pred, [ref]), 0.5)

    def test_threshold_boundary_below(self):
        # Construct NL just above threshold so score becomes 0.
        # For equal-length strings, NL = substitutions/len.
        # 6 substitutions => NL=0.6 => sim=0.4 < 0.5 => 0
        pred = "aaaaaaaaaa"
        ref2 = "aaaabbbbbb"
        self.assertEqual(anls_single(pred, [ref2]), 0.0)

    def test_threshold_constant(self):
        self.assertEqual(ANLS_THRESHOLD, 0.5)

    def test_multiple_refs_max(self):
        # One exact ref should yield 1 even if others mismatch.
        self.assertEqual(anls_single("nnk", ["zzzz", "NNK"]), 1.0)

    def test_numeric_near(self):
        # "83.4%" vs "83.4 %." — after normalize: "83.4%" vs "83.4 %."
        # spaces collapsed but period remains; should be high similarity.
        score = anls_single("83.4 %.", ["83.4%"])
        self.assertGreater(score, 0.5)

    def test_failure_zeros(self):
        m = score_example(None, ["a"], status="error")
        self.assertEqual(m["anls"], 0.0)
        self.assertTrue(m["failure"])


class TestAggregate(unittest.TestCase):
    def test_requested_includes_failures(self):
        rows = [
            {
                "question_id": 1,
                "status": "ok",
                "prediction": "a",
                "reference_answers": ["a"],
            },
            {
                "question_id": 2,
                "status": "error",
                "prediction": None,
                "reference_answers": ["a"],
            },
        ]
        # Score each
        for r in rows:
            r["metrics"] = score_example(
                r.get("prediction"), r["reference_answers"], status=r["status"]
            )
        agg = aggregate_scores(rows, n_requested=2)
        self.assertEqual(agg["n_completed_ok"], 1)
        self.assertEqual(agg["n_failed"], 1)
        self.assertAlmostEqual(agg["mean_anls_requested"], 0.5)
        self.assertAlmostEqual(agg["mean_anls_completed_ok_only"], 1.0)


if __name__ == "__main__":
    unittest.main()
