"""InternVL3 adapter: official dynamic 448px tiling + generate()-scoped timing.

Preprocessing follows the Transformers inference recipe on the model card for
``OpenGVLab/InternVL3-1B`` (dynamic_preprocess, max_num=12, use_thumbnail=True).
This is **not** a matched visual-token budget against SmolVLM.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from vlm_doc.model import count_total_parameters

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_INSTRUCTION = (
    "Answer the question using a single word or short phrase."
)


@dataclass
class LoadedInternVL:
    model: Any
    tokenizer: Any
    device: torch.device
    dtype: torch.dtype
    revision: str
    model_id: str
    total_parameters: int
    bf16_supported: bool
    image_size: int
    max_num: int
    use_thumbnail: bool
    num_image_token_per_tile: int
    template_name: str
    architecture_notes: dict[str, Any]


def build_transform(input_size: int):
    mean, std = IMAGENET_MEAN, IMAGENET_STD
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize(
                (input_size, input_size), interpolation=InterpolationMode.BICUBIC
            ),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image: Image.Image,
    *,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> list[Image.Image]:
    """Official InternVL dynamic tiling (model-card recipe)."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images: list[Image.Image] = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_pixel_values(
    image: Image.Image,
    *,
    input_size: int = 448,
    max_num: int = 12,
    use_thumbnail: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    transform = build_transform(input_size)
    tiles = dynamic_preprocess(
        image,
        image_size=input_size,
        use_thumbnail=use_thumbnail,
        max_num=max_num,
    )
    pixel_values = torch.stack([transform(t) for t in tiles])
    info = {
        "tile_count": int(pixel_values.shape[0]),
        "image_size": input_size,
        "max_num": max_num,
        "use_thumbnail": use_thumbnail,
        "thumbnail_appended": bool(
            use_thumbnail and len(tiles) > 1
        ),  # True when thumbnail path taken
        "original_size_wh": list(image.size),
        "pixel_values_shape": list(pixel_values.shape),
        "recipe": "official_internvl_dynamic_preprocess_max_num12_thumbnail",
    }
    return pixel_values, info


def load_internvl(
    model_id: str,
    revision: str,
    *,
    device: torch.device | None = None,
    prefer_bf16: bool = True,
    use_flash_attn: bool = False,
    max_num: int = 12,
    use_thumbnail: bool = True,
) -> LoadedInternVL:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16_supported = (
        bool(torch.cuda.is_bf16_supported()) if device.type == "cuda" else False
    )
    use_bf16 = prefer_bf16 and device.type == "cuda" and bf16_supported
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, trust_remote_code=True, use_fast=False
    )
    model = AutoModel.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=use_flash_attn,
    )
    model = model.to(device)
    model.eval()

    total = count_total_parameters(model)
    if total >= 1_000_000_000:
        raise AssertionError(
            f"InternVL total unique parameters {total} exceeds 1e9 budget"
        )

    image_size = int(getattr(model.config, "force_image_size", 448) or 448)
    num_image_token = int(getattr(model, "num_image_token", 0) or 0)
    template = str(getattr(model, "template", getattr(model.config, "template", "")))
    notes = {
        "architectures": list(getattr(model.config, "architectures", []) or []),
        "vision": "InternVisionModel (config.vision_config)",
        "language": str(
            getattr(getattr(model.config, "llm_config", None), "architectures", None)
        ),
        "downsample_ratio": getattr(model.config, "downsample_ratio", None),
        "max_dynamic_patch": getattr(model.config, "max_dynamic_patch", None),
        "use_thumbnail_config": getattr(model.config, "use_thumbnail", None),
        "dynamic_image_size": getattr(model.config, "dynamic_image_size", None),
        "use_flash_attn_requested": use_flash_attn,
        "license_hint": "apache-2.0 (Hub tag)",
        "hub_url": f"https://huggingface.co/{model_id}",
        "training_data_disclosure": (
            "Hub tags reference OpenGVLab/MMPR-v1.2 and base_model "
            "InternVL3-1B-Instruct; full training mixture details treated as unknown "
            "beyond public model-card disclosures."
        ),
    }

    return LoadedInternVL(
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype if device.type == "cuda" else torch.float32,
        revision=revision,
        model_id=model_id,
        total_parameters=total,
        bf16_supported=bf16_supported,
        image_size=image_size,
        max_num=int(max_num),
        use_thumbnail=bool(use_thumbnail),
        num_image_token_per_tile=num_image_token,
        template_name=template,
        architecture_notes=notes,
    )


def _build_query_and_ids(
    loaded: LoadedInternVL,
    *,
    question_with_image_marker: str,
    num_patches: int,
) -> tuple[str, torch.Tensor, torch.Tensor, int]:
    """Mirror InternVLChatModel.chat prompt construction without carrying history."""
    from . import internvl_conv  # local import of get_conv_template wrapper

    IMG_START_TOKEN = "<img>"
    IMG_END_TOKEN = "</img>"
    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

    model = loaded.model
    tokenizer = loaded.tokenizer
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id

    template = internvl_conv.get_conv_template(
        loaded.template_name or model.template, model=model
    )
    template.system_message = model.system_message
    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

    template.append_message(template.roles[0], question_with_image_marker)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    image_tokens = (
        IMG_START_TOKEN
        + IMG_CONTEXT_TOKEN * loaded.num_image_token_per_tile * num_patches
        + IMG_END_TOKEN
    )
    query = query.replace("<image>", image_tokens, 1)

    model_inputs = tokenizer(query, return_tensors="pt")
    input_ids = model_inputs["input_ids"].to(loaded.device)
    attention_mask = model_inputs["attention_mask"].to(loaded.device)
    return query, input_ids, attention_mask, int(eos_token_id)


def run_one_internvl(
    loaded: LoadedInternVL,
    *,
    image: Image.Image,
    question: str,
    max_new_tokens: int = 64,
    instruction: str = DEFAULT_INSTRUCTION,
) -> dict[str, Any]:
    """Single-turn InternVL inference; times model.generate() only."""
    text = f"{question.strip()}\n{instruction}"
    question_marked = f"<image>\n{text}"

    pixel_values, tile_info = load_pixel_values(
        image,
        input_size=loaded.image_size,
        max_num=loaded.max_num,
        use_thumbnail=loaded.use_thumbnail,
    )
    pixel_values = pixel_values.to(device=loaded.device, dtype=loaded.dtype)
    num_patches = int(pixel_values.shape[0])
    visual_token_count = int(num_patches * loaded.num_image_token_per_tile)
    tile_info["visual_token_count"] = visual_token_count
    tile_info["num_image_token_per_tile"] = loaded.num_image_token_per_tile

    query, input_ids, attention_mask, eos_token_id = _build_query_and_ids(
        loaded,
        question_with_image_marker=question_marked,
        num_patches=num_patches,
    )
    input_len = int(input_ids.shape[-1])

    gen_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
        "eos_token_id": eos_token_id,
    }

    if loaded.device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.inference_mode():
        generation_output = loaded.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )
    if loaded.device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    gpu_mem: dict[str, Any] = {}
    if loaded.device.type == "cuda":
        gpu_mem = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "scope": (
                "torch.cuda.max_memory_{allocated,reserved} since reset_peak_memory_stats "
                "immediately before this generate() call"
            ),
        }

    out_ids = generation_output[0]
    out_len = int(out_ids.numel())
    # Verified (Phase 2D probe job 82527 + transformers 4.48.3 source):
    # InternVLChatModel.generate passes inputs_embeds (not input_ids) to
    # language_model.generate; HF initializes starter input_ids as (batch, 0);
    # returned LongTensor is newly generated token IDs only. Official chat
    # batch_decodes the full return with no input_len slice. Evidence artifact:
    # results/internvl_generate_return_probe.json
    if out_len > input_len and torch.equal(out_ids[:input_len], input_ids[0]):
        new_tokens = out_ids[input_len:]
        return_convention = "prompt_plus_generated"
    else:
        new_tokens = out_ids
        return_convention = "generated_only"

    prediction_decoded_full = loaded.tokenizer.decode(
        new_tokens, skip_special_tokens=True
    )
    prediction_raw = prediction_decoded_full
    # Match InternVL chat postprocess: split on conversation sep if present.
    from . import internvl_conv

    template = internvl_conv.get_conv_template(
        loaded.template_name or loaded.model.template, model=loaded.model
    )
    sep = template.sep.strip()
    if sep and sep in prediction_raw:
        prediction_raw = prediction_raw.split(sep)[0]
    prediction = prediction_raw.strip()
    gen_len = int(new_tokens.numel())
    max_new = int(max_new_tokens)

    return {
        "prediction": prediction,
        "prediction_decoded_full": prediction_decoded_full,
        "prediction_raw": prediction_raw,
        "prediction_stripped": prediction,
        "input_token_length": input_len,
        "generated_token_length": gen_len,
        "generation_cap_reached": gen_len >= max_new,
        "generation_seconds": t1 - t0,
        "timing_scope": "model.generate_only_cuda_synchronized",
        "timing_comparable_to_smolvlm_generate": True,
        "generate_return_convention": return_convention,
        "image_info": tile_info,
        "instruction": instruction,
        "max_new_tokens": max_new,
        "do_sample": False,
        "dtype": str(loaded.dtype).replace("torch.", ""),
        "gpu_memory": gpu_mem,
        "adapter": "internvl3",
        "query_has_image_marker": "<image>" in question_marked,
        "prompt_echo_check": (
            "fail" if question.strip() and question.strip() in prediction else "ok"
        ),
    }


def processor_settings_dict(loaded: LoadedInternVL) -> dict[str, Any]:
    return {
        "adapter": "internvl3",
        "image_size": loaded.image_size,
        "max_num": loaded.max_num,
        "use_thumbnail": loaded.use_thumbnail,
        "num_image_token_per_tile": loaded.num_image_token_per_tile,
        "template": loaded.template_name,
        "dynamic_image_size": True,
        "recipe": "official_dynamic_preprocess_448_max_num12_thumbnail",
        "architecture_notes": loaded.architecture_notes,
        "not_matched_visual_token_budget_vs_smolvlm": True,
    }
