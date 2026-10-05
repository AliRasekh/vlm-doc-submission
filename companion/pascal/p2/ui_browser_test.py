#!/usr/bin/env python3
"""Interactive Gradio UI test: Playwright browser + Gradio Client callbacks."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _ensure_paths() -> None:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    sys.path.insert(0, str(REPO_ROOT / "companion" / "pascal"))


def _client_actions(url: str, out: dict) -> None:
    """Gradio Client callbacks (requires schema patch in this process too)."""
    from p2.gradio_schema_patch import apply_gradio_schema_patch
    from p2.presets import get_preset

    apply_gradio_schema_patch()
    from gradio_client import Client, handle_file

    exact = get_preset("exact_success")
    fail = get_preset("failure")
    client = Client(url)
    out["gradio_client"] = {"connected": True}

    preset_out = client.predict(exact["label"], api_name="/apply_preset")
    out["gradio_client"]["apply_preset"] = {
        "question": preset_out[1] if isinstance(preset_out, (list, tuple)) else None,
        "status": preset_out[-1] if isinstance(preset_out, (list, tuple)) else str(preset_out),
    }
    out["actions"].append("client_apply_preset")

    r12 = client.predict(
        handle_file(exact["image_path"]),
        exact["question"],
        12,
        api_name="/run_live",
    )
    out["gradio_client"]["cap12"] = {
        "answer": r12[0],
        "meta": r12[1],
        "status": r12[5],
        "refs": r12[4],
    }
    out["actions"].append("client_run_cap12")

    r4 = client.predict(
        handle_file(exact["image_path"]),
        exact["question"],
        4,
        api_name="/run_live",
    )
    out["gradio_client"]["cap4"] = {
        "answer": r4[0],
        "meta": r4[1],
        "status": r4[5],
    }
    out["actions"].append("client_run_cap4")

    t12 = re.search(r"actual_tile_count=(\d+)", str(r12[1] or ""))
    t4 = re.search(r"actual_tile_count=(\d+)", str(r4[1] or ""))
    out["tile_counts"] = {
        "cap12": int(t12.group(1)) if t12 else None,
        "cap4": int(t4.group(1)) if t4 else None,
    }

    ru = client.predict(
        handle_file(fail["image_path"]),
        fail["question"],
        12,
        api_name="/run_live",
    )
    out["gradio_client"]["upload_like"] = {
        "answer": ru[0],
        "status": ru[5],
        "refs": ru[4],
    }
    out["actions"].append("client_upload_like_run")

    bad = client.predict(
        handle_file(exact["image_path"]),
        "   ",
        12,
        api_name="/run_live",
    )
    out["gradio_client"]["invalid"] = {"status": bad[5], "answer": bad[0]}
    out["actions"].append("client_invalid_empty_question")

    errs = []
    if str(out["gradio_client"]["cap12"].get("status", "")).strip() != "ok":
        errs.append("client cap12 status not ok")
    if not str(out["gradio_client"]["cap12"].get("answer") or "").strip():
        errs.append("client cap12 empty answer")
    if str(out["gradio_client"]["cap4"].get("status", "")).strip() != "ok":
        errs.append("client cap4 status not ok")
    if (
        out["tile_counts"]["cap12"] is not None
        and out["tile_counts"]["cap4"] is not None
        and out["tile_counts"]["cap12"] == out["tile_counts"]["cap4"]
    ):
        errs.append("client tile counts not isolated")
    inv_status = str((out["gradio_client"].get("invalid") or {}).get("status") or "")
    if "Error" not in inv_status:
        errs.append(f"client invalid input did not error: {inv_status!r}")
    # Unrelated upload must not keep exact_success refs as the only content;
    # refs update is a gr.update dict/string — just ensure upload answered.
    if not str(out["gradio_client"]["upload_like"].get("answer") or "").strip():
        errs.append("client upload-like empty answer")
    out["gradio_client"]["errors"] = errs
    out["gradio_client"]["ok"] = len(errs) == 0


def _playwright_actions(url: str, out: dict, shot_dir: Path, timeout_ms: int) -> None:
    from playwright.sync_api import sync_playwright

    from p2.presets import get_preset

    exact = get_preset("exact_success")
    fail = get_preset("failure")

    def shot(page, name: str) -> None:
        path = shot_dir / f"{name}.png"
        page.screenshot(path=str(path), full_page=True)
        out["screenshots"].append(str(path))

    def read_labeled(page, label: str) -> str:
        loc = page.get_by_label(label)
        if loc.count() == 0:
            return ""
        try:
            return loc.input_value()
        except Exception:
            try:
                return loc.inner_text()
            except Exception:
                return ""

    def wait_new_result(
        page, prev_metrics: str, *, timeout_s: float = 180.0, expect_error: bool = False
    ) -> tuple[str, str]:
        """Wait until metrics change (new inference) or status becomes Error."""
        deadline = time.time() + timeout_s
        last_status = ""
        last_metrics = prev_metrics
        while time.time() < deadline:
            last_status = read_labeled(page, "Status / errors")
            last_metrics = read_labeled(page, "Request metrics")
            if last_status.strip().startswith("Error"):
                return last_status, last_metrics
            if (
                not expect_error
                and last_status.strip() == "ok"
                and last_metrics.strip()
                and last_metrics != prev_metrics
            ):
                return last_status, last_metrics
            time.sleep(0.5)
        return last_status, last_metrics

    def select_preset(page, label_substr: str) -> bool:
        # Gradio 5 Dropdown: the interactive hit target is usually .secondary-wrap.
        block = page.locator("div.block").filter(
            has_text="Development presets (illustrative)"
        ).first
        block.scroll_into_view_if_needed()
        for sel in (".secondary-wrap", ".wrap-inner", "[role='listbox']", "input"):
            ctl = block.locator(sel).first
            if ctl.count() == 0:
                continue
            try:
                ctl.click(timeout=5_000, force=True)
                break
            except Exception:
                continue
        else:
            return False
        page.wait_for_timeout(700)
        option = page.locator(
            "[role='listbox'] [role='option'], ul.options li, .options li, li"
        ).filter(has_text=label_substr)
        if option.count() == 0:
            option = page.get_by_text(label_substr, exact=False)
        if option.count() == 0:
            return False
        option.first.click(timeout=10_000, force=True)
        page.wait_for_timeout(400)
        return True

    def set_cap(page, value: str) -> None:
        # Prefer native radio input; fall back to visible label text.
        radio = page.locator(f"input[type='radio'][value='{value}']")
        if radio.count():
            radio.first.check(force=True)
        else:
            block = page.locator("div.block").filter(has_text="max_num").first
            block.get_by_text(value, exact=True).first.click(force=True)
        page.wait_for_timeout(400)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1400})
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        page.wait_for_timeout(3500)
        shot(page, "01_open")
        out["actions"].append("browser_open_page")
        out["browser_title"] = page.title()

        preset_ok = select_preset(page, "Exact success")
        out["browser_preset_dropdown_ok"] = preset_ok
        if preset_ok:
            page.get_by_role("button", name="Load preset fields").click()
            page.wait_for_timeout(2500)
            out["actions"].append("browser_load_preset")
        q_after = read_labeled(page, "Question")
        out["browser_preset_question"] = q_after
        if not q_after.strip():
            # Fallback: populate via upload widget + question (still real UI widgets)
            page.locator("input[type='file']").first.set_input_files(exact["image_path"])
            page.get_by_label("Question").fill(exact["question"])
            page.wait_for_timeout(1000)
            out["actions"].append("browser_manual_fill_preset_fields")
        shot(page, "02_preset_loaded")

        set_cap(page, "12")
        prev = read_labeled(page, "Request metrics")
        page.get_by_role("button", name="Run", exact=True).click()
        st, met = wait_new_result(page, prev)
        shot(page, "03_run_cap12")
        out["browser_cap12"] = {
            "status": st,
            "answer": read_labeled(page, "Model response"),
            "metrics": met,
        }
        out["actions"].append("browser_run_cap12")

        set_cap(page, "4")
        prev = met
        page.get_by_role("button", name="Run", exact=True).click()
        st4, met4 = wait_new_result(page, prev)
        shot(page, "04_run_cap4")
        out["browser_cap4"] = {
            "status": st4,
            "answer": read_labeled(page, "Model response"),
            "metrics": met4,
        }
        out["actions"].append("browser_run_cap4")

        page.locator("input[type='file']").first.set_input_files(fail["image_path"])
        page.wait_for_timeout(1500)
        q = page.get_by_label("Question")
        q.click()
        q.fill("")
        q.fill(fail["question"])
        page.wait_for_timeout(800)
        prev = met4
        page.get_by_role("button", name="Run", exact=True).click()
        stu, metu = wait_new_result(page, prev)
        shot(page, "05_upload_run")
        out["browser_upload"] = {
            "status": stu,
            "answer": read_labeled(page, "Model response"),
            "metrics": metu,
        }
        out["actions"].append("browser_upload_run")

        q = page.get_by_label("Question")
        q.click()
        q.fill("")
        q.fill("   ")
        page.wait_for_timeout(400)
        prev_status = read_labeled(page, "Status / errors")
        page.get_by_role("button", name="Run", exact=True).click()
        deadline = time.time() + 60
        stb = prev_status
        while time.time() < deadline:
            stb = read_labeled(page, "Status / errors")
            if stb.strip().startswith("Error"):
                break
            time.sleep(0.5)
        shot(page, "06_invalid_empty_question")
        out["browser_invalid"] = {"status": stb}
        out["actions"].append("browser_invalid_empty_question")

        browser_errs = []
        if (out.get("browser_cap12") or {}).get("status", "").strip() != "ok":
            browser_errs.append("browser cap12 not ok")
        if not (out.get("browser_cap12") or {}).get("answer", "").strip():
            browser_errs.append("browser cap12 empty answer")
        if (out.get("browser_cap4") or {}).get("status", "").strip() != "ok":
            browser_errs.append("browser cap4 not ok")
        m12 = (out.get("browser_cap12") or {}).get("metrics") or ""
        m4 = (out.get("browser_cap4") or {}).get("metrics") or ""
        t12 = re.search(r"actual_tile_count=(\d+)", m12)
        t4 = re.search(r"actual_tile_count=(\d+)", m4)
        c12 = re.search(r"cap=(\d+)", m12)
        c4 = re.search(r"cap=(\d+)", m4)
        out["browser_tile_counts"] = {
            "cap12": int(t12.group(1)) if t12 else None,
            "cap4": int(t4.group(1)) if t4 else None,
            "cap_field_12": int(c12.group(1)) if c12 else None,
            "cap_field_4": int(c4.group(1)) if c4 else None,
        }
        if out["browser_tile_counts"]["cap_field_4"] not in (4, None):
            if out["browser_tile_counts"]["cap_field_4"] == 12:
                browser_errs.append("browser cap4 run still reports cap=12")
        if (
            out["browser_tile_counts"]["cap12"]
            and out["browser_tile_counts"]["cap4"]
            and out["browser_tile_counts"]["cap12"]
            == out["browser_tile_counts"]["cap4"]
            and out["browser_tile_counts"]["cap_field_4"] == 4
        ):
            browser_errs.append("browser tile counts not isolated at cap4")
        if "Error" not in (stb or ""):
            browser_errs.append(f"browser invalid did not error: {stb!r}")
        if len(out["screenshots"]) < 4:
            browser_errs.append("insufficient screenshots")
        out["browser_errors"] = browser_errs
        out["browser_ok"] = len(browser_errs) == 0
        browser.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--screenshot-dir", required=True)
    parser.add_argument("--timeout-ms", type=int, default=180_000)
    args = parser.parse_args()

    _ensure_paths()

    out: dict = {
        "task": "pascal_p2_1_browser_ui",
        "url": args.url,
        "recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "performed": False,
        "ok": False,
        "actions": [],
        "screenshots": [],
        "errors": [],
    }
    shot_dir = Path(args.screenshot_dir)
    shot_dir.mkdir(parents=True, exist_ok=True)

    try:
        _client_actions(args.url, out)
    except Exception as exc:  # noqa: BLE001
        out["gradio_client"] = {"ok": False, "error": str(exc)}
        out["errors"].append(f"gradio_client: {exc}")

    try:
        _playwright_actions(args.url, out, shot_dir, args.timeout_ms)
        out["performed"] = True
    except ImportError as exc:
        out["browser_blocker"] = f"playwright not installed: {exc}"
        out["errors"].append(out["browser_blocker"])
        out["browser_ok"] = False
    except Exception as exc:  # noqa: BLE001
        out["browser_blocker"] = str(exc)
        out["errors"].append(f"playwright: {exc}")
        out["browser_ok"] = False
        out["performed"] = bool(out.get("screenshots"))

    client_ok = bool((out.get("gradio_client") or {}).get("ok"))
    browser_ok = bool(out.get("browser_ok"))
    out["ok"] = browser_ok and client_ok
    out["visual_qa_passed"] = browser_ok
    out["client_api_ok"] = client_ok
    if not browser_ok and client_ok:
        out["note"] = (
            "Gradio Client API interactions succeeded; browser visual QA did not. "
            "Do not label visual QA as passed."
        )

    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(
        json.dumps(
            {
                "ok": out["ok"],
                "visual_qa_passed": out.get("visual_qa_passed"),
                "client_api_ok": out.get("client_api_ok"),
                "actions": out.get("actions"),
                "errors": out.get("errors"),
                "browser_errors": out.get("browser_errors"),
                "tile_counts": out.get("tile_counts"),
                "n_screenshots": len(out.get("screenshots") or []),
            },
            indent=2,
        )
    )
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
