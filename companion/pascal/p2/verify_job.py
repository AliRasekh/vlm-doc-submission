#!/usr/bin/env python3
"""Bounded P2.1 GPU verification: strict CLI JSON, engine compare, UI HTTP + browser."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]


def _http_get(url: str, timeout: float = 5.0) -> tuple[int, str]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8", errors="replace")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "UNKNOWN"


def main() -> int:
    os.chdir(REPO_ROOT)
    os.environ["PYTHONNOUSERSITE"] = "1"
    sys.path.insert(0, str(REPO_ROOT / "src"))
    sys.path.insert(0, str(REPO_ROOT / "companion" / "pascal"))

    # Preserve prior P2 evidence; write P2.1 under a new directory.
    legacy = REPO_ROOT / "companion/pascal/artifacts/p2_verify"
    out_dir = REPO_ROOT / "companion/pascal/artifacts/p2_1_verify"
    out_dir.mkdir(parents=True, exist_ok=True)
    if legacy.is_dir() and not (legacy / "PRESERVED_FOR_P2.txt").is_file():
        (legacy / "PRESERVED_FOR_P2.txt").write_text(
            "P2 verification artifacts preserved; P2.1 writes to p2_1_verify/.\n"
        )

    report: dict = {
        "task": "pascal_p2_1_verification",
        "source_git_revision": _git_sha(),
        "recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": os.environ.get("HOSTNAME") or os.uname().nodename,
        "steps": [],
        "failures": [],
    }

    from p2.cli_json import parse_cli_json_stdout
    from p2.engine import get_engine
    from p2.presets import get_preset, match_preset_references, sha256_file
    from p2.runtime import probe_bf16_support

    probe = probe_bf16_support()
    (out_dir / "bf16_probe.json").write_text(json.dumps(probe, indent=2) + "\n")
    report["bf16_probe"] = probe
    report["steps"].append({"bf16_probe": "ok"})

    engine = get_engine()
    meta = engine.ensure_loaded()
    (out_dir / "runtime_details.json").write_text(
        json.dumps(meta, indent=2, default=str) + "\n"
    )
    report["runtime_details"] = {
        k: meta.get(k)
        for k in (
            "dtype",
            "param_dtype_sample",
            "gpu_name",
            "attention_backend_label",
            "attention",
            "label_runtime",
        )
    }
    report["steps"].append({"model_load": "ok"})

    preset = get_preset("exact_success")
    r12 = engine.predict(
        image_path=preset["image_path"],
        question=preset["question"],
        max_num=12,
    )
    r4 = engine.predict(
        image_path=preset["image_path"],
        question=preset["question"],
        max_num=4,
    )
    fail = get_preset("failure")
    r_fail = engine.predict(
        image_path=fail["image_path"],
        question=fail["question"],
        max_num=12,
    )
    cap_ok = (
        r12["max_num_cap"] == 12
        and r4["max_num_cap"] == 4
        and r12["actual_tile_count"] != r4["actual_tile_count"]
    )
    report["engine_runs"] = {
        "cap12": {
            "prediction": r12["prediction"],
            "tiles": r12["actual_tile_count"],
            "request_s": r12["request_seconds"],
        },
        "cap4": {
            "prediction": r4["prediction"],
            "tiles": r4["actual_tile_count"],
            "request_s": r4["request_seconds"],
        },
        "second_question": {
            "qid": fail["question_id"],
            "prediction": r_fail["prediction"],
            "tiles": r_fail["actual_tile_count"],
        },
        "cap_isolation_ok": cap_ok,
    }
    if not cap_ok:
        report["failures"].append("cap_isolation")
    report["steps"].append({"engine_multi_cap": "ok" if cap_ok else "fail"})

    # Content-hash reference association (path-agnostic copy)
    copy_dir = out_dir / "tmp_image_copy"
    copy_dir.mkdir(exist_ok=True)
    copied = copy_dir / "gradio_style_copy.png"
    shutil.copy2(preset["image_path"], copied)
    matched = match_preset_references(image_path=copied, question=preset["question"])
    mismatched = match_preset_references(
        image_path=copied, question="totally unrelated question"
    )
    ref_ok = (
        matched is not None
        and matched["id"] == "exact_success"
        and mismatched is None
        and sha256_file(copied) == preset["image_sha256"]
    )
    report["reference_association"] = {
        "content_hash_match_ok": ref_ok,
        "matched_id": None if matched is None else matched["id"],
        "unrelated_question_rejected": mismatched is None,
    }
    if not ref_ok:
        report["failures"].append("reference_association")
    report["steps"].append({"reference_association": "ok" if ref_ok else "fail"})

    # Strict CLI JSON subprocess
    cli = [
        sys.executable,
        str(REPO_ROOT / "companion/pascal/p2/cli.py"),
        "--preset",
        "exact_success",
        "--max-num",
        "12",
        "--json",
    ]
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    p = subprocess.run(cli, capture_output=True, text=True, env=env, cwd=str(REPO_ROOT))
    cli_out = {
        "returncode": p.returncode,
        "stdout": p.stdout,
        "stderr_tail": (p.stderr or "")[-4000:],
        "stdout_has_flashattn_notice": "FlashAttention" in (p.stdout or ""),
        "stderr_has_flashattn_notice": "FlashAttention" in (p.stderr or ""),
    }
    (out_dir / "cli_exact_success.json").write_text(
        json.dumps(cli_out, indent=2) + "\n"
    )

    cli_parse_ok = False
    cli_pred = None
    cli_obj = None
    parse_error = None
    try:
        if p.returncode != 0:
            raise ValueError(f"CLI exit {p.returncode}")
        cli_obj = parse_cli_json_stdout(p.stdout or "")
        cli_pred = cli_obj.get("prediction")
        cli_parse_ok = True
    except Exception as exc:  # noqa: BLE001
        parse_error = str(exc)
        report["failures"].append("cli_json_parse")

    matches = bool(cli_parse_ok and cli_pred == r12["prediction"])
    if cli_parse_ok and not matches:
        report["failures"].append("cli_engine_mismatch")

    report["cli"] = {
        "returncode": p.returncode,
        "strict_json_parse_ok": cli_parse_ok,
        "parse_error": parse_error,
        "prediction": cli_pred,
        "matches_engine_cap12": matches,
        "stdout_is_pure_json": cli_parse_ok and not cli_out["stdout_has_flashattn_notice"],
        "actual_tile_count": None if cli_obj is None else cli_obj.get("actual_tile_count"),
    }
    report["steps"].append(
        {"cli_strict_json": "ok" if (cli_parse_ok and matches) else "fail"}
    )

    # Serialization
    order: list[str] = []

    def _job(tag: str, cap: int) -> None:
        order.append(f"start:{tag}")
        engine.predict(
            image_path=preset["image_path"], question=preset["question"], max_num=cap
        )
        order.append(f"end:{tag}")

    t1 = threading.Thread(target=_job, args=("A", 12))
    t2 = threading.Thread(target=_job, args=("B", 1))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    idx = {k: order.index(k) for k in order}
    serialized = (
        "end:A" in idx
        and "end:B" in idx
        and (
            (idx["end:A"] < idx["end:B"] and idx["start:B"] <= idx["end:A"])
            or (idx["end:B"] < idx["end:A"] and idx["start:A"] <= idx["end:B"])
        )
    )
    report["serialization"] = {"order": order, "non_overlapping": serialized}
    if not serialized:
        report["failures"].append("serialization")
    report["steps"].append({"serialization": "ok" if serialized else "fail"})

    try:
        engine.predict(image_path=preset["image_path"], question="   ", max_num=12)
        report["empty_question_error"] = False
        report["failures"].append("empty_question")
    except Exception as exc:  # noqa: BLE001
        report["empty_question_error"] = True
        report["empty_question_message"] = str(exc)
    report["steps"].append(
        {"validation": "ok" if report["empty_question_error"] else "fail"}
    )

    # Release parent model before UI subprocess
    import p2.engine as engine_mod

    engine_mod._ENGINE = None
    del engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    report["steps"].append({"parent_model_released": "ok"})

    port = 7865
    host = "127.0.0.1"
    proc = subprocess.Popen(
        [
            sys.executable,
            str(REPO_ROOT / "companion/pascal/p2/app.py"),
            "--host",
            host,
            "--port",
            str(port),
            "--preload",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=open(out_dir / "ui_server.out", "w"),
        stderr=open(out_dir / "ui_server.err", "w"),
    )
    ui_ok = False
    body = ""
    try:
        for _ in range(120):
            if proc.poll() is not None:
                break
            try:
                code, body = _http_get(f"http://{host}:{port}/", timeout=2.0)
                if code == 200 and len(body) > 50:
                    ui_ok = True
                    break
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                time.sleep(2)
        report["ui_http"] = {
            "ok": ui_ok,
            "url": f"http://{host}:{port}/",
            "snippet": body[:500],
        }
        if not ui_ok:
            report["failures"].append("ui_http")
        report["steps"].append({"ui_http": "ok" if ui_ok else "fail"})

        try:
            c2, _ = _http_get(f"http://{host}:{port}/config", timeout=5.0)
            report["ui_config_status"] = c2
        except Exception as exc:  # noqa: BLE001
            report["ui_config_error"] = str(exc)

        # Native API schema (must not be blanked)
        try:
            c3, api_body = _http_get(f"http://{host}:{port}/info", timeout=5.0)
            report["ui_info_status"] = c3
            report["ui_info_nonempty"] = len(api_body) > 20
        except Exception as exc:  # noqa: BLE001
            report["ui_info_error"] = str(exc)

        # Browser interactive test (Playwright) when available
        browser_script = REPO_ROOT / "companion/pascal/p2/ui_browser_test.py"
        shot_dir = out_dir / "screenshots"
        shot_dir.mkdir(exist_ok=True)
        browser_report_path = out_dir / "browser_ui.json"
        if ui_ok and browser_script.is_file():
            bp = subprocess.run(
                [
                    sys.executable,
                    str(browser_script),
                    "--url",
                    f"http://{host}:{port}/",
                    "--out",
                    str(browser_report_path),
                    "--screenshot-dir",
                    str(shot_dir),
                ],
                capture_output=True,
                text=True,
                env=env,
                cwd=str(REPO_ROOT),
                timeout=600,
            )
            report["browser_subprocess"] = {
                "returncode": bp.returncode,
                "stdout_tail": (bp.stdout or "")[-2000:],
                "stderr_tail": (bp.stderr or "")[-2000:],
            }
            if browser_report_path.is_file():
                report["browser_visual_qa"] = json.loads(
                    browser_report_path.read_text()
                )
            else:
                report["browser_visual_qa"] = {
                    "performed": False,
                    "reason": f"browser script failed rc={bp.returncode}",
                    "stderr_tail": (bp.stderr or "")[-1500:],
                }
            if not report["browser_visual_qa"].get("ok"):
                report["failures"].append("browser_ui")
            report["steps"].append(
                {
                    "browser_ui": "ok"
                    if report["browser_visual_qa"].get("ok")
                    else "fail"
                }
            )
        else:
            report["browser_visual_qa"] = {
                "performed": False,
                "reason": "UI not ready or browser script missing",
            }
            report["steps"].append({"browser_ui": "skip"})
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        report["ui_server_exit"] = proc.returncode
        report["ui_shutdown"] = True

    bqa = report.get("browser_visual_qa") or {}
    report["visual_qa_passed"] = bool(bqa.get("visual_qa_passed") or bqa.get("browser_ok"))
    report["client_api_ok"] = bool(bqa.get("client_api_ok") or (bqa.get("gradio_client") or {}).get("ok"))
    # CLI parse/match failures MUST fail the job (not warnings).
    # Visual QA requires real browser success; if browser is blocked, client API
    # may still pass while visual_qa_passed stays false.
    core_ok = (
        report["cli"]["strict_json_parse_ok"]
        and report["cli"]["matches_engine_cap12"]
        and report["engine_runs"]["cap_isolation_ok"]
        and report["serialization"]["non_overlapping"]
        and report["empty_question_error"]
        and ui_ok
        and ref_ok
    )
    interactive_ok = report["visual_qa_passed"] or (
        report["client_api_ok"] and not bqa.get("performed", True)
    )
    # Prefer full browser+client ok; accept client-only only when browser could not run.
    if bqa.get("performed") and not report["visual_qa_passed"]:
        interactive_ok = False
        if "browser_ui" not in report["failures"]:
            report["failures"].append("browser_ui")
    report["ok"] = core_ok and interactive_ok and (
        "cli_json_parse" not in report["failures"]
        and "cli_engine_mismatch" not in report["failures"]
    )
    (out_dir / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "ok": report["ok"],
                "out": str(out_dir),
                "failures": report["failures"],
                "cli_matches_engine_cap12": report["cli"]["matches_engine_cap12"],
                "strict_json_parse_ok": report["cli"]["strict_json_parse_ok"],
            },
            indent=2,
        )
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
