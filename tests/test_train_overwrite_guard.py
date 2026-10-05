"""Training overwrite protection must fail before model loading."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_internvl_lora.py"


class TestTrainOverwriteGuard(unittest.TestCase):
    def test_refuses_existing_outputs_before_cuda_or_model_load(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            log_path = tmp / "train.jsonl"
            summary_path = tmp / "train_summary.json"
            ckpt_root = tmp / "checkpoints"
            ex_md = tmp / "example.md"
            ex_json = tmp / "example.json"
            log_path.write_text("{}\n")
            summary_path.write_text("{}\n")
            (ckpt_root / "update_0100").mkdir(parents=True)
            (ckpt_root / "update_0100" / "adapter_config.json").write_text("{}\n")
            ex_md.write_text("# existing\n")
            ex_json.write_text("{}\n")

            cfg = tmp / "train_cfg.yaml"
            cfg.write_text(
                textwrap.dedent(
                    f"""\
                    seed: 0
                    log_jsonl: {log_path}
                    summary_json: {summary_path}
                    supervised_example_md: {ex_md}
                    supervised_example_json: {ex_json}
                    checkpoint_root: {ckpt_root}
                    save_updates: [100, 200]
                    model_id: OpenGVLab/InternVL3-1B
                    revision: 4415a3b810e636d11dfa86b0e9ba40bb00535aa8
                    """
                )
            )

            env = os.environ.copy()
            env["PYTHONNOUSERSITE"] = "1"

            proc = subprocess.run(
                [sys.executable, str(TRAIN_SCRIPT), "--config", str(cfg)],
                cwd=str(REPO_ROOT),
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(proc.returncode, 2, msg=proc.stderr)
            self.assertIn("refusing to overwrite existing training outputs", proc.stderr)
            self.assertIn(str(log_path), proc.stderr)
            self.assertIn(str(summary_path), proc.stderr)
            self.assertIn(str(ckpt_root / "update_0100"), proc.stderr)
            # Must not reach the CUDA-required branch (which runs after the guard).
            self.assertNotIn("CUDA required for training", proc.stderr)
            # Existing files must remain untouched.
            self.assertEqual(log_path.read_text(), "{}\n")
            self.assertEqual(summary_path.read_text(), "{}\n")


if __name__ == "__main__":
    unittest.main()
