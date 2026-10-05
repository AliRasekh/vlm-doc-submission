"""Regression tests for supervised assistant termination (Phase 3C)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vlm_doc.train.collator import (  # noqa: E402
    SupervisedBoundaryError,
    assert_supervised_termination,
    resolve_assistant_termination,
)


class TestResolveAssistantTermination(unittest.TestCase):
    def test_template_emits_eos_then_newline_keeps_single_eos(self):
        # prompt | answer | eos | newline  --> truncate to prompt|answer|eos
        eos = 151645
        nl = 198
        prompt = torch.tensor([1, 2, 3, eos, nl, 10, 11], dtype=torch.long)  # user eos ok
        answer = torch.tensor([20, 21, 22], dtype=torch.long)
        full = torch.cat([prompt, answer, torch.tensor([eos, nl])])
        out = resolve_assistant_termination(full, prompt_len=prompt.numel(), eos_id=eos)
        self.assertEqual(out.tolist(), prompt.tolist() + answer.tolist() + [eos])
        self.assertEqual(int(out[-1]), eos)
        self.assertNotEqual(int(out[-1]), nl)

    def test_exactly_one_supervised_eos_at_end(self):
        eos = 99
        prompt_len = 4
        full = torch.tensor([1, 2, 3, 4, 50, 51, eos, 7, eos], dtype=torch.long)
        # First eos in assistant suffix is at relative index 2; trailing junk dropped.
        out = resolve_assistant_termination(full, prompt_len=prompt_len, eos_id=eos)
        self.assertEqual(out.tolist(), [1, 2, 3, 4, 50, 51, eos])
        labels = out.clone()
        labels[:prompt_len] = -100
        assert_supervised_termination(labels, prompt_len=prompt_len, eos_id=eos)

    def test_no_supervised_tokens_after_terminator(self):
        eos = 5
        prompt_len = 2
        full = torch.tensor([1, 2, 8, 9, eos, 3], dtype=torch.long)
        out = resolve_assistant_termination(full, prompt_len=prompt_len, eos_id=eos)
        labels = out.clone()
        labels[:prompt_len] = -100
        assert_supervised_termination(labels, prompt_len=prompt_len, eos_id=eos)
        self.assertEqual(labels.tolist(), [-100, -100, 8, 9, eos])

    def test_answer_tokens_intact_before_eos(self):
        eos = 7
        prompt = torch.tensor([1, 1, 1], dtype=torch.long)
        answer = torch.tensor([42, 43, 44], dtype=torch.long)
        full = torch.cat([prompt, answer, torch.tensor([eos, 0])])
        out = resolve_assistant_termination(full, prompt_len=3, eos_id=eos)
        self.assertEqual(out[3:-1].tolist(), answer.tolist())

    def test_missing_eos_appends_exactly_one(self):
        eos = 77
        full = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        out = resolve_assistant_termination(full, prompt_len=3, eos_id=eos)
        self.assertEqual(out.tolist(), [1, 2, 3, 4, 5, eos])
        labels = out.clone()
        labels[:3] = -100
        assert_supervised_termination(labels, prompt_len=3, eos_id=eos)

    def test_rejects_empty_assistant(self):
        with self.assertRaises(SupervisedBoundaryError):
            resolve_assistant_termination(
                torch.tensor([1, 2, 3]), prompt_len=3, eos_id=9
            )

    def test_rejects_masked_prompt_leak(self):
        labels = torch.tensor([1, -100, 2, 9], dtype=torch.long)
        with self.assertRaises(SupervisedBoundaryError):
            assert_supervised_termination(labels, prompt_len=2, eos_id=9)

    def test_rejects_double_supervised_eos(self):
        labels = torch.tensor([-100, -100, 1, 9, 2, 9], dtype=torch.long)
        with self.assertRaises(SupervisedBoundaryError):
            assert_supervised_termination(labels, prompt_len=2, eos_id=9)

    def test_does_not_use_user_turn_eos_as_boundary(self):
        # User eos appears before prompt_len; assistant has its own eos+newline.
        eos = 151645
        # [user_toks..., user_eos, nl, assistant_hdr..., answer..., asst_eos, nl]
        full = torch.tensor(
            [10, 11, eos, 198, 20, 21, 30, 31, eos, 198], dtype=torch.long
        )
        prompt_len = 6  # includes through assistant header
        out = resolve_assistant_termination(full, prompt_len=prompt_len, eos_id=eos)
        self.assertEqual(out.tolist(), [10, 11, eos, 198, 20, 21, 30, 31, eos])


@unittest.skipUnless(
    (ROOT / ".hf_cache/hub/models--OpenGVLab--InternVL3-1B").exists(),
    "pinned InternVL tokenizer cache required",
)
class TestRealTokenizerTermination(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        snap = (
            ROOT
            / ".hf_cache/hub/models--OpenGVLab--InternVL3-1B/snapshots"
            / "4415a3b810e636d11dfa86b0e9ba40bb00535aa8"
        )
        sys.path.insert(0, str(snap))
        from conversation import get_conv_template  # type: ignore

        cls.tok = AutoTokenizer.from_pretrained(
            "OpenGVLab/InternVL3-1B",
            trust_remote_code=True,
            revision="4415a3b810e636d11dfa86b0e9ba40bb00535aa8",
            use_fast=False,
        )
        cls._get_conv_template = staticmethod(get_conv_template)
        cls.eos_id = cls.tok.convert_tokens_to_ids("<|im_end|>")

    def _build(self, answer: str, question: str = "Q?"):
        img = "<img>" + ("<IMG_CONTEXT>" * 8) + "</img>"
        qm = f"<image>\n{question}\nAnswer the question using a single word or short phrase."
        qm = qm.replace("<image>", img, 1)
        tp = self._get_conv_template("internvl2_5")
        tp.system_message = "SYS"
        tp.append_message(tp.roles[0], qm)
        tp.append_message(tp.roles[1], None)
        tf = self._get_conv_template("internvl2_5")
        tf.system_message = "SYS"
        tf.append_message(tf.roles[0], qm)
        tf.append_message(tf.roles[1], answer)
        prompt_ids = self.tok(tp.get_prompt(), return_tensors="pt")["input_ids"][0]
        full_ids = self.tok(tf.get_prompt(), return_tensors="pt")["input_ids"][0]
        self.assertTrue(torch.equal(full_ids[: prompt_ids.numel()], prompt_ids))
        # Historical bug pattern: template ends with eos+newline.
        self.assertEqual(
            self.tok.convert_ids_to_tokens(full_ids[-2:].tolist()),
            ["<|im_end|>", "Ċ"],
        )
        fixed = resolve_assistant_termination(
            full_ids, prompt_len=int(prompt_ids.numel()), eos_id=self.eos_id
        )
        labels = fixed.clone()
        labels[: prompt_ids.numel()] = -100
        assert_supervised_termination(
            labels, prompt_len=int(prompt_ids.numel()), eos_id=self.eos_id
        )
        return prompt_ids, full_ids, fixed, labels

    def test_qid_46350_style_long_answer(self):
        answer = (
            "a post-hoc analysis of VMI data has been performed and will be "
            "included as part of the submission."
        )
        _p, raw, fixed, labels = self._build(answer)
        # Before (buggy append) would be ... eos, nl, eos
        buggy = torch.cat([raw, torch.tensor([self.eos_id], dtype=raw.dtype)])
        self.assertEqual(
            self.tok.convert_ids_to_tokens(buggy[-3:].tolist()),
            ["<|im_end|>", "Ċ", "<|im_end|>"],
        )
        self.assertEqual(
            self.tok.convert_ids_to_tokens(fixed[-3:].tolist())[-1],
            "<|im_end|>",
        )
        self.assertNotIn("Ċ", self.tok.convert_ids_to_tokens(labels[labels != -100].tolist()))

    def test_short_and_numeric_answers(self):
        for ans in ("Mg", "117", "T-Th-S", "981261"):
            _p, _raw, fixed, labels = self._build(ans)
            sup = labels[labels != -100].tolist()
            self.assertEqual(sup[-1], self.eos_id)
            # Answer tokens remain before eos (not truncated away).
            self.assertGreaterEqual(len(sup), 2)
            decoded = self.tok.decode(sup[:-1], skip_special_tokens=False)
            # Decode may add spaces; answer chars must remain.
            for ch in ans.replace(" ", ""):
                self.assertIn(ch, decoded.replace(" ", ""))


if __name__ == "__main__":
    unittest.main()
