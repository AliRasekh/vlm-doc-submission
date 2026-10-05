"""CPU tests for manifest hash and inference fingerprint invalidation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vlm_doc.run_fingerprint import (
    assert_output_fingerprint_coherent,
    build_inference_fingerprint,
    canonical_manifest_content_hash,
    verify_manifest_hash,
)


class TestManifestHash(unittest.TestCase):
    def test_self_hash_excluded_and_mismatch_detected(self):
        body = {"name": "x", "question_ids": [1, 2], "examples": []}
        h = canonical_manifest_content_hash(body)
        manifest = {**body, "manifest_sha256": h}
        self.assertTrue(verify_manifest_hash(manifest)["ok"])
        bad = {**manifest, "question_ids": [1, 2, 3]}
        # stored hash no longer matches content
        self.assertFalse(verify_manifest_hash(bad)["ok"])


class TestFingerprint(unittest.TestCase):
    def _fp(self, **over):
        base = dict(
            model_id="m",
            model_revision="r",
            manifest_content_sha256="h",
            instruction="i",
            max_new_tokens=64,
            do_sample=False,
            prefer_bf16=True,
            seed=0,
            config_path="c.yaml",
            config_sha256="c",
            resolved_dtype="bfloat16",
            processor_settings={"size": 512},
            software={"torch": "2.5.1"},
            inference_source_hashes={"src/vlm_doc/infer.py": "abc"},
        )
        base.update(over)
        return build_inference_fingerprint(**base)

    def test_inference_code_hash_changes_fingerprint(self):
        a = self._fp(inference_source_hashes={"src/vlm_doc/infer.py": "aaa"})
        b = self._fp(inference_source_hashes={"src/vlm_doc/infer.py": "bbb"})
        self.assertNotEqual(a["run_fingerprint_sha256"], b["run_fingerprint_sha256"])

    def test_docs_not_in_fingerprint_payload(self):
        a = self._fp()
        self.assertNotIn("README.md", a["inference_source_hashes"])

    def test_mixed_output_rejected(self):
        fp = self._fp()["run_fingerprint_sha256"]
        other = self._fp(seed=1)["run_fingerprint_sha256"]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "out.jsonl"
            with p.open("w") as f:
                f.write(json.dumps({"question_id": 1, "run_fingerprint": {"run_fingerprint_sha256": fp}}) + "\n")
                f.write(json.dumps({"question_id": 2, "run_fingerprint": {"run_fingerprint_sha256": other}}) + "\n")
            with self.assertRaises(RuntimeError):
                assert_output_fingerprint_coherent(p, fp)


if __name__ == "__main__":
    unittest.main()
