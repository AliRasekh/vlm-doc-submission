#!/usr/bin/env python3
"""Export a full source-review packet by concatenating actual on-disk files.

Does not manually summarize source; embeds file contents with delimiters.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def git_meta(root: Path) -> dict[str, str]:
    def run(args: list[str]) -> str:
        try:
            return subprocess.check_output(
                args, cwd=root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception as exc:  # noqa: BLE001
            return f"UNAVAILABLE: {exc}"

    dirty = run(["git", "status", "--porcelain"])
    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": "dirty" if dirty else "clean",
        "dirty_detail": dirty if dirty else "(clean)",
        "remote_head": run(["git", "rev-parse", "origin/main"]),
    }


def resolve(root: Path, rel: str) -> Path | None:
    p = (root / rel).resolve()
    if p.is_file():
        return p
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--label",
        default="phase02b",
        help="Packet label used in header",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    os.chdir(root)

    # Required / expected inventory (resolve actual paths; state missing explicitly)
    groups: list[tuple[str, list[str]]] = [
        (
            "CORE_METRICS_SCORING",
            [
                "src/vlm_doc/metrics.py",
                "scripts/rescore_predictions.py",
                "tests/test_metrics.py",
                "tests/test_metrics_v2.py",
            ],
        ),
        (
            "FINGERPRINT_RESUME_INFERENCE",
            [
                "src/vlm_doc/run_fingerprint.py",
                "src/vlm_doc/model.py",
                "src/vlm_doc/infer.py",
                "src/vlm_doc/metadata.py",
                "scripts/run_eval.py",
                "tests/test_fingerprint.py",
            ],
        ),
        (
            "SPLIT_AUDIT_HOLDOUT_V2",
            [
                "scripts/audit_split_leakage.py",
                "scripts/rebuild_holdout_v2.py",
                "scripts/prepare_frozen_splits.py",
            ],
        ),
        (
            "FORMATTING_DIAGNOSTICS_GALLERY",
            [
                "scripts/analyze_phase02b_cpu_followup.py",
                "scripts/compare_dev_256_500.py",
                "docs/examples/gallery_tables_generated.md",
                "docs/examples/README.md",
                "docs/progress/phase02b_cpu_followup.md",
            ],
        ),
        (
            "CONFIGS_AND_SLURM",
            [
                "configs/models/smolvlm_256m.yaml",
                "configs/models/smolvlm_500m.yaml",
                "scripts/slurm/dev_smolvlm256m_neumann.sbatch",
                "scripts/slurm/dev_smolvlm500m_neumann.sbatch",
                "scripts/slurm/smoke_neumann.sbatch",
                "scripts/check_env_isolation.py",
            ],
        ),
        (
            "LEAKAGE_AUDIT_OUTPUTS",
            [
                "results/split_leakage_audit_phase02b.json",
                "results/split_leakage_holdout_v1.json",
                "results/split_leakage_holdout_v2.json",
            ],
        ),
        (
            "AGGREGATE_SUMMARIES_AND_DIAGNOSTICS",
            [
                "results/historical/dev_smolvlm256m_summary_phase02a.json",
                "results/dev_smolvlm256m_rescore_strict_v2.json",
                "results/dev_smolvlm256m_rescore_inclusive_v1.json",
                "results/dev_smolvlm500m_summary.json",
                "results/dev_smolvlm500m_smoke4_summary.json",
                "results/dev_256_vs_500_comparison.json",
                "results/phase02b_cpu_followup_analysis.json",
            ],
        ),
        (
            "SOURCE_REFERENCE_DOCS",
            [
                "docs/sources.md",
                "docs/decisions.md",
                "docs/progress/phase02b.md",
                "README.md",
                "data/README.md",
                "src/vlm_doc/README.md",
                "scripts/README.md",
            ],
        ),
        (
            "OPTIONAL_PHASE2C_IF_PRESENT",
            [
                "configs/models/internvl3_1b.yaml",
                "src/vlm_doc/adapters/internvl.py",
                "src/vlm_doc/adapters/internvl_conv.py",
                "src/vlm_doc/adapters/__init__.py",
                "scripts/slurm/dev_internvl3_1b_neumann.sbatch",
                "scripts/compare_dev_three_models.py",
                "scripts/export_full_review_packet.py",
                "results/dev_internvl3_1b_summary.json",
                "results/dev_internvl3_1b_smoke4_summary.json",
                "results/dev_three_model_comparison.json",
                "results/model_download_internvl3_1b.json",
                "docs/progress/phase02c.md",
            ],
        ),
        (
            "PHASE2D_DECODE_TRAIN_SUBSET",
            [
                "scripts/probe_internvl_generate_return.py",
                "scripts/slurm/probe_internvl_generate_return.sbatch",
                "scripts/prepare_train_subset_v1.py",
                "scripts/analyze_internvl_error_sample.py",
                "data/manifests/docvqa_train_subset_v1.json",
                "results/internvl_generate_return_probe.json",
                "results/train_subset_v1_leakage_audit.json",
                "results/internvl_error_sample_v1.json",
                "docs/progress/phase02d.md",
                "docs/progress/phase02d_error_sample.md",
                "docs/examples/train_subset_v1_example.md",
            ],
        ),
        (
            "PHASE3A_LORA",
            [
                "configs/train/internvl3_1b_lora_v1.yaml",
                "src/vlm_doc/train/__init__.py",
                "src/vlm_doc/train/lora.py",
                "src/vlm_doc/train/collator.py",
                "src/vlm_doc/adapters/internvl.py",
                "src/vlm_doc/run_fingerprint.py",
                "scripts/audit_train_subset_images_v1.py",
                "scripts/gate_internvl_lora.py",
                "scripts/train_internvl_lora.py",
                "scripts/compare_internvl_lora_dev.py",
                "scripts/run_eval.py",
                "scripts/slurm/audit_train_subset_images.sbatch",
                "scripts/slurm/phase03a_internvl_lora.sbatch",
                "scripts/export_full_review_packet.py",
                "environment/requirements.txt",
                "environment/versions.verified.txt",
                "results/train_subset_v1_image_integrity_audit.json",
                "results/internvl_lora_gate_ok.json",
                "results/train_internvl3_1b_lora_v1_summary.json",
                "results/train_internvl3_1b_lora_v1_curves.json",
                "results/dev_internvl3_1b_lora_u100_summary.json",
                "results/dev_internvl3_1b_lora_u200_summary.json",
                "results/dev_internvl_lora_v1_comparison.json",
                "results/train_lora_v1_supervised_example.json",
                "docs/progress/phase03a.md",
                "docs/examples/train_lora_v1_supervised_example.md",
                "docs/examples/lora_v1_before_after.md",
                "docs/decisions.md",
                "README.md",
            ],
        ),
        (
            "PHASE3B_ROBUSTNESS_PROTOCOL",
            [
                "src/vlm_doc/image_perturbations.py",
                "src/vlm_doc/svg_plot.py",
                "src/vlm_doc/run_fingerprint.py",
                "scripts/audit_phase03a_evidence.py",
                "scripts/prepare_robustness_subset_v1.py",
                "scripts/run_robustness_dev_v1.py",
                "scripts/analyze_robustness_v1.py",
                "scripts/bootstrap_holdout_anls_diff.py",
                "scripts/slurm/phase03b_robustness.sbatch",
                "scripts/slurm/final_holdout_eval.sbatch",
                "scripts/export_full_review_packet.py",
                "data/manifests/docvqa_dev_robustness_v1.json",
                "results/phase03a_evidence_audit.json",
                "results/robustness_v1_analysis.json",
                "results/robustness_v1/run_index.json",
                "results/robustness_v1/internvl3_1b_base__clean_summary.json",
                "results/robustness_v1/internvl3_1b_base__half_detail_summary.json",
                "results/robustness_v1/internvl3_1b_base__jpeg_q40_summary.json",
                "results/robustness_v1/internvl3_1b_lora_u100__clean_summary.json",
                "results/robustness_v1/internvl3_1b_lora_u100__half_detail_summary.json",
                "results/robustness_v1/internvl3_1b_lora_u100__jpeg_q40_summary.json",
                "docs/progress/phase03b.md",
                "docs/evaluation/final_protocol.md",
                "docs/examples/phase03a_train_loss_curve.svg",
                "docs/examples/phase03a_eight_changed.md",
                "docs/examples/robustness_v1_gallery.md",
                "docs/examples/robustness_v1_anls.svg",
                "docs/decisions.md",
                "docs/sources.md",
                "README.md",
            ],
        ),
        (
            "PHASE3C_TERMINATION_FIX",
            [
                "src/vlm_doc/train/collator.py",
                "src/vlm_doc/train/__init__.py",
                "src/vlm_doc/train/lora.py",
                "tests/test_collator_termination.py",
                "configs/train/internvl3_1b_lora_v2.yaml",
                "scripts/train_internvl_lora.py",
                "scripts/gate_internvl_lora.py",
                "scripts/audit_supervised_termination_v1.py",
                "scripts/compare_internvl_lora_dev.py",
                "scripts/run_robustness_dev_v1.py",
                "scripts/slurm/phase03c_internvl_lora_v2.sbatch",
                "scripts/slurm/final_holdout_eval.sbatch",
                "scripts/export_full_review_packet.py",
                "results/supervised_termination_audit_v1.json",
                "results/supervised_termination_qid46350_before_after.json",
                "results/train_lora_v1_supervised_example.json",
                "results/train_lora_v2_supervised_example.json",
                "results/internvl_lora_v2_gate_ok.json",
                "results/train_internvl3_1b_lora_v2_summary.json",
                "results/train_internvl3_1b_lora_v2_curves.json",
                "results/dev_internvl3_1b_lora_v2_u100_summary.json",
                "results/dev_internvl3_1b_lora_v2_u200_summary.json",
                "results/dev_internvl_lora_v2_comparison.json",
                "results/phase03c_v1_vs_v2_dev.json",
                "results/phase03c_v1_log_guard.json",
                "results/robustness_v2_analysis.json",
                "results/robustness_v2/run_index.json",
                "results/robustness_v2/internvl3_1b_lora_v2_u200__clean_summary.json",
                "results/robustness_v2/internvl3_1b_lora_v2_u200__half_detail_summary.json",
                "results/robustness_v2/internvl3_1b_lora_v2_u200__jpeg_q40_summary.json",
                "docs/progress/phase03c.md",
                "docs/evaluation/final_protocol.md",
                "docs/examples/train_lora_v2_supervised_example.md",
                "docs/examples/lora_v2_before_after.md",
                "docs/examples/phase03c_train_loss_curve_v2.svg",
                "docs/decisions.md",
                "docs/sources.md",
                "README.md",
            ],
        ),
        (
            "PHASE4_HOLDOUT_REPORT",
            [
                "docs/evaluation/final_protocol.md",
                "results/phase04_evaluation_revision.json",
                "results/phase04_preflight.json",
                "results/phase04_packaging_revision.json",
                "scripts/slurm/final_holdout_eval.sbatch",
                "scripts/analyze_holdout_final.py",
                "scripts/bootstrap_holdout_anls_diff.py",
                "scripts/export_full_review_packet.py",
                "results/holdout_final_table.json",
                "results/holdout_final_analysis.json",
                "results/holdout_internvl_anls_bootstrap.json",
                "results/holdout_smolvlm256m_summary.json",
                "results/holdout_smolvlm500m_summary.json",
                "results/holdout_internvl3_1b_summary.json",
                "results/holdout_internvl3_1b_lora_v2_u200_summary.json",
                "docs/progress/phase04.md",
                "docs/report/evidence_index.md",
                "docs/report/limitations.md",
                "docs/report/holdout_gallery.md",
                "docs/report/figures/holdout_primary_anls.svg",
                "docs/report/figures/architecture_lora_placement.svg",
                "docs/report/figures/pipeline_flow.svg",
                "docs/handoffs/pascal_companion.md",
                "docs/decisions.md",
                "docs/sources.md",
                "README.md",
            ],
        ),
    ]

    meta = git_meta(root)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    inventory: list[dict] = []
    missing: list[str] = []

    with out_path.open("w", encoding="utf-8") as out:
        out.write(f"# FULL SOURCE REVIEW PACKET ({args.label})\n")
        out.write(f"# export_timestamp_utc={ts}\n")
        out.write(f"# git_commit={meta['commit']}\n")
        out.write(f"# git_branch={meta['branch']}\n")
        out.write(f"# git_dirty_state={meta['dirty']}\n")
        out.write(f"# origin_main={meta['remote_head']}\n")
        out.write(f"# repo_root={root}\n")
        out.write(
            "# NOTE: Full file contents follow. Hashes alone are insufficient; "
            "each section embeds the on-disk source.\n"
        )
        out.write("# EXCLUDED: secrets, credentials, weights, image bytes, caches, full datasets.\n")
        out.write("\n")
        out.write("## GIT_DIRTY_DETAIL\n")
        out.write(meta["dirty_detail"] + "\n\n")

        for group_name, rels in groups:
            out.write(f"\n{'=' * 80}\n")
            out.write(f"## GROUP: {group_name}\n")
            out.write(f"{'=' * 80}\n\n")
            for rel in rels:
                path = resolve(root, rel)
                if path is None:
                    missing.append(rel)
                    out.write(f"### MISSING_REQUIRED_OR_OPTIONAL: {rel}\n")
                    out.write("STATUS: FILE NOT FOUND AT RESOLVED PATH\n\n")
                    inventory.append(
                        {"path": rel, "status": "missing", "bytes": 0}
                    )
                    continue
                data = path.read_bytes()
                # Skip huge optional dumps
                if len(data) > 8_000_000:
                    out.write(f"### SKIPPED_TOO_LARGE: {rel} bytes={len(data)}\n\n")
                    inventory.append(
                        {
                            "path": str(path.relative_to(root)),
                            "status": "skipped_too_large",
                            "bytes": len(data),
                        }
                    )
                    continue
                try:
                    text = data.decode("utf-8")
                except UnicodeDecodeError:
                    out.write(
                        f"### SKIPPED_BINARY: {rel} bytes={len(data)}\n\n"
                    )
                    inventory.append(
                        {
                            "path": str(path.relative_to(root)),
                            "status": "skipped_binary",
                            "bytes": len(data),
                        }
                    )
                    continue
                rel_out = str(path.relative_to(root))
                inventory.append(
                    {"path": rel_out, "status": "included", "bytes": len(data)}
                )
                out.write(f"{'~' * 80}\n")
                out.write(f"BEGIN_FILE path={rel_out} abs={path} bytes={len(data)}\n")
                out.write(f"{'~' * 80}\n")
                out.write(text)
                if not text.endswith("\n"):
                    out.write("\n")
                out.write(f"{'~' * 80}\n")
                out.write(f"END_FILE path={rel_out}\n")
                out.write(f"{'~' * 80}\n\n")

        out.write(f"\n{'=' * 80}\n")
        out.write("## FILE_INVENTORY\n")
        out.write(f"{'=' * 80}\n")
        total = 0
        n_inc = 0
        for item in inventory:
            out.write(
                f"{item['status']:18s}  {item['bytes']:10d}  {item['path']}\n"
            )
            if item["status"] == "included":
                total += item["bytes"]
                n_inc += 1
        out.write(f"\nincluded_files={n_inc}\n")
        out.write(f"included_source_bytes={total}\n")
        out.write(f"missing_count={len(missing)}\n")
        for m in missing:
            out.write(f"missing={m}\n")

    size = out_path.stat().st_size
    print(f"REVIEW_PACKET_ABS_PATH={out_path}")
    print(f"REVIEW_PACKET_BYTES={size}")
    print(f"REVIEW_PACKET_MIB={size / (1024 * 1024):.3f}")
    print(f"INCLUDED_FILES={sum(1 for i in inventory if i['status']=='included')}")
    print(f"MISSING_COUNT={len(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
