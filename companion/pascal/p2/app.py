#!/usr/bin/env python3
"""P2 Gradio UI — loopback only; no public share/tunnels."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def build_demo():
    # Apply targeted schema patch before Gradio builds event schemas.
    from p2.gradio_schema_patch import apply_gradio_schema_patch

    patch_info = apply_gradio_schema_patch()

    import gradio as gr

    from p2.engine import get_engine
    from p2.presets import get_preset, list_presets, load_prerecorded_p1, match_preset_references
    from p2.validate import ValidationError, validate_cap

    engine = get_engine()
    presets = list_presets()
    preset_labels = {p["label"]: p["id"] for p in presets}

    def _details_md() -> str:
        d = engine.details()
        if not d.get("ready"):
            return (
                "_Model not loaded yet. First Run will load once per process._  \n"
                f"_Gradio schema patch: `{patch_info}`_"
            )
        probe = d.get("bf16_probe") or {}
        return (
            f"**Model:** `{d.get('model_id')}@{d.get('revision')}`  \n"
            f"**GPU:** `{d.get('gpu_name')}`  \n"
            f"**dtype:** `{d.get('dtype')}` (param sample `{d.get('param_dtype_sample')}`)  \n"
            f"**Attention:** `{d.get('attention_backend_label')}` "
            f"(flash_attn={d.get('use_flash_attn')})  \n"
            f"**BF16 default support bool:** `{probe.get('is_bf16_supported_default')}`  \n"
            f"**BF16 including_emulation=False:** "
            f"`{probe.get('is_bf16_supported_including_emulation_false')}`  \n"
            f"**Compute capability:** `{probe.get('compute_capability')}`  \n"
            f"**Runtime label:** {d.get('label_runtime')}  \n"
            f"**Memory:** omitted in live UI (no P1 reset monkey-patch).  \n"
            f"**Gradio schema patch:** `{patch_info.get('applied')}`"
        )

    def on_load():
        try:
            engine.ensure_loaded()
            return _details_md(), "Model loaded."
        except Exception as exc:  # noqa: BLE001
            return _details_md(), f"Load error: {exc}"

    def apply_preset(label: str):
        pid = preset_labels.get(label)
        if not pid:
            return None, "", 12, "Select a preset.", gr.update(visible=False, value=""), ""
        p = get_preset(pid)
        note = (
            f"Preset `{p['id']}` (qid {p['question_id']}). "
            "Illustrative development example — not a representative accuracy sample. "
            "References stay hidden until after Run."
        )
        return (
            p["image_path"],
            p["question"],
            int(p["default_cap"]),
            note,
            gr.update(visible=False, value=""),
            "Preset fields loaded (references cleared until Run).",
        )

    def clear_refs_on_change():
        """Clear stale reference panel when image/question changes."""
        return gr.update(visible=False, value="")

    def run_live(image, question, max_num):
        cleanup = None
        if image is None:
            return (
                "",
                "",
                "",
                _details_md(),
                gr.update(visible=False, value=""),
                "Error: no image provided.",
            )
        image_path = image if isinstance(image, str) else getattr(image, "name", None)
        if image_path is None and hasattr(image, "save"):
            from p2.engine import save_upload_bytes
            import io

            buf = io.BytesIO()
            image.save(buf, format="PNG")
            image_path = str(save_upload_bytes(buf.getvalue(), suffix=".png"))
            cleanup = image_path

        try:
            result = engine.predict(
                image_path=image_path,
                question=question,
                max_num=int(max_num),
            )
            meta = (
                f"cap={result['max_num_cap']} ( "
                f"{'default' if result['max_num_cap']==12 else 'reduced visual budget'} )\n"
                f"actual_tile_count={result['actual_tile_count']}\n"
                f"request_seconds={result['request_seconds']:.4f}\n"
                f"timing_scope={result['request_timing_scope']}\n"
                f"live_inference={result['live_inference']}\n"
                f"references_used_in_model={result['references_used_in_model']}"
            )
            matched = match_preset_references(
                image_path=image_path, question=str(question or "")
            )
            if matched:
                refs = matched.get("references") or []
                refs_md = (
                    "**Reference answers (dev scoring sidecar; not used by model)**\n\n"
                    f"_matched preset `{matched['id']}` via image SHA256 + question_\n\n"
                    + ("\n".join(f"- `{r}`" for r in refs) if refs else "_none_")
                )
                refs_update = gr.update(visible=True, value=refs_md)
            else:
                refs_update = gr.update(visible=False, value="")
            return (
                result.get("prediction") or "",
                meta,
                json.dumps(result, indent=2, default=str),
                _details_md(),
                refs_update,
                "ok",
            )
        except ValidationError as exc:
            return (
                "",
                "",
                "",
                _details_md(),
                gr.update(visible=False, value=""),
                f"Error: {exc}",
            )
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc(limit=4)
            return (
                "",
                "",
                "",
                _details_md(),
                gr.update(visible=False, value=""),
                f"Error: {exc}\n{tb}",
            )
        finally:
            if cleanup:
                try:
                    Path(cleanup).unlink(missing_ok=True)
                except OSError:
                    pass

    def show_prerecorded(label: str, max_num: int):
        pid = preset_labels.get(label)
        if not pid:
            return "Select a preset first."
        p = get_preset(pid)
        try:
            cap = validate_cap(max_num)
        except ValidationError as exc:
            return str(exc)
        rec = load_prerecorded_p1(int(p["question_id"]), cap)
        if rec is None:
            return "No prerecorded P1 row found for this preset/cap (artifact missing)."
        return (
            f"**PRERECORDED — not live inference**\n\n"
            f"qid={rec['question_id']} cap={rec['max_num_cap']}\n"
            f"prediction: `{rec.get('prediction')}`\n"
            f"tiles: {rec.get('actual_tile_count')}\n"
            f"source: `{rec.get('source')}`"
        )

    with gr.Blocks(title="Pascal P2 InternVL DocVQA Demo") as demo:
        gr.Markdown(
            "# Pascal P2 — InternVL document QA (supplementary demo)\n"
            "Pinned `OpenGVLab/InternVL3-1B` base. Frozen instruction; greedy; "
            "`max_new_tokens=64`. **No holdout. No LoRA.** Loopback only."
        )
        with gr.Row():
            with gr.Column():
                image = gr.Image(
                    type="filepath",
                    label="Document image (upload or preset path)",
                )
                question = gr.Textbox(label="Question", lines=2)
                gr.Markdown(
                    "**Tile cap (`max_num`):** `12` = default visual budget; "
                    "`4` / `1` = reduced visual budgets."
                )
                max_num = gr.Radio(
                    choices=[12, 4, 1],
                    value=12,
                    label="max_num",
                )
                preset = gr.Dropdown(
                    choices=list(preset_labels.keys()),
                    label="Development presets (illustrative)",
                    value=None,
                )
                preset_note = gr.Markdown("")
                with gr.Row():
                    btn_preset = gr.Button("Load preset fields")
                    btn_run = gr.Button("Run", variant="primary")
                    btn_load = gr.Button("Load model now")
                status = gr.Textbox(label="Status / errors", interactive=False)
            with gr.Column():
                answer = gr.Textbox(label="Model response", lines=3)
                meta = gr.Textbox(label="Request metrics", lines=6)
                details = gr.Markdown(_details_md())
                refs = gr.Markdown(visible=False)
                raw = gr.Textbox(label="Raw JSON (live)", lines=8)
                gr.Markdown("### Offline P1 viewer (prerecorded)")
                gr.Markdown("Shows **recorded P1 outputs** only. Not live inference.")
                btn_offline = gr.Button("Show prerecorded for preset + cap")
                offline_out = gr.Markdown("")

        btn_load.click(on_load, outputs=[details, status], api_name="on_load")
        btn_preset.click(
            apply_preset,
            inputs=[preset],
            outputs=[image, question, max_num, preset_note, refs, status],
            api_name="apply_preset",
        )
        # Clear stale references when inputs change (including Gradio-cached copies).
        image.change(clear_refs_on_change, outputs=[refs], api_name="clear_refs_image")
        question.change(
            clear_refs_on_change, outputs=[refs], api_name="clear_refs_question"
        )
        btn_run.click(
            run_live,
            inputs=[image, question, max_num],
            outputs=[answer, meta, raw, details, refs, status],
            api_name="run_live",
        )
        btn_offline.click(
            show_prerecorded,
            inputs=[preset, max_num],
            outputs=[offline_out],
            api_name="show_prerecorded",
        )
    return demo


def main() -> int:
    os.chdir(REPO_ROOT)
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    os.environ.setdefault("GRADIO_SERVER_NAME", "127.0.0.1")
    sys.path.insert(0, str(REPO_ROOT / "src"))
    sys.path.insert(0, str(REPO_ROOT / "companion" / "pascal"))

    from p2.gradio_schema_patch import apply_gradio_schema_patch

    apply_gradio_schema_patch()

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Load model before accepting UI traffic",
    )
    args = parser.parse_args()

    if args.host not in ("127.0.0.1", "localhost"):
        print(
            "ERROR: bind host must be loopback (127.0.0.1). Refusing public bind.",
            file=sys.stderr,
        )
        return 2

    demo = build_demo()
    # Verify native get_api_info works after targeted patch (do not blank schema).
    try:
        info = demo.get_api_info()
        print(
            f"gradio_get_api_info_ok keys={list(info)[:6] if isinstance(info, dict) else type(info)}",
            file=sys.stderr,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: get_api_info still failing after patch: {exc}", file=sys.stderr)
        return 3

    if args.preload:
        from p2.engine import get_engine

        print("Preloading model…", file=sys.stderr)
        print(json.dumps(get_engine().ensure_loaded(), indent=2, default=str), file=sys.stderr)

    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
        show_error=True,
        inbrowser=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
