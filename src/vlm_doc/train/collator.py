"""Supervised InternVL multimodal example builder (teacher-forced labels)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from vlm_doc.adapters.internvl import (
    DEFAULT_INSTRUCTION,
    LoadedInternVL,
    load_pixel_values,
)
from vlm_doc.adapters import internvl_conv

IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


@dataclass
class SupervisedExample:
    question_id: int
    question: str
    target_answer: str
    input_ids: torch.Tensor  # [1, S]
    attention_mask: torch.Tensor
    labels: torch.Tensor
    pixel_values: torch.Tensor  # [P, 3, H, W]
    image_flags: torch.Tensor  # [P]
    num_patches: int
    seq_len: int
    n_supervised_tokens: int
    n_img_context: int
    prompt_len: int
    assistant_eos_offset: int  # index of supervised eos within full sequence


class SupervisedBoundaryError(ValueError):
    """Invalid assistant termination boundary for supervised labels."""


def _expand_image_tokens(query: str, *, num_patches: int, num_image_token: int) -> str:
    image_tokens = (
        IMG_START_TOKEN
        + IMG_CONTEXT_TOKEN * (num_image_token * num_patches)
        + IMG_END_TOKEN
    )
    if "<image>" not in query:
        raise ValueError("prompt missing <image> marker")
    return query.replace("<image>", image_tokens, 1)


def resolve_assistant_termination(
    full_ids: torch.Tensor,
    *,
    prompt_len: int,
    eos_id: int,
) -> torch.Tensor:
    """Truncate to answer + exactly one assistant end-of-turn token.

    The conversation template may emit ``<|im_end|>\\n`` after the assistant
    message. Searching for the first eos in the *full* conversation is wrong
    because user/system turns also contain end tokens. The terminator must be
    located inside the assistant suffix (tokens after the inference prompt).
    """
    if prompt_len < 0 or prompt_len > int(full_ids.numel()):
        raise SupervisedBoundaryError(
            f"invalid prompt_len={prompt_len} for seq_len={full_ids.numel()}"
        )
    assistant = full_ids[prompt_len:]
    if assistant.numel() == 0:
        raise SupervisedBoundaryError("empty assistant suffix after prompt")

    eos_pos = (assistant == int(eos_id)).nonzero(as_tuple=False).view(-1)
    if eos_pos.numel() == 0:
        # Template did not emit an end token; append exactly one.
        return torch.cat(
            [full_ids, torch.tensor([int(eos_id)], dtype=full_ids.dtype)], dim=0
        )

    # First eos in the assistant suffix is the intended terminator.
    end = int(prompt_len) + int(eos_pos[0].item()) + 1
    truncated = full_ids[:end]
    if int(truncated[-1].item()) != int(eos_id):
        raise SupervisedBoundaryError("truncation failed to end on eos")
    # Reject if anything after that first assistant eos would have been kept
    # (defensive: we sliced to end inclusive).
    if truncated.numel() <= prompt_len:
        raise SupervisedBoundaryError("no supervised tokens after truncation")
    return truncated


def assert_supervised_termination(
    labels: torch.Tensor,
    *,
    prompt_len: int,
    eos_id: int,
    question_id: int = -1,
) -> None:
    """Validate answer-only supervision with exactly one terminal eos."""
    lab = labels.view(-1)
    if prompt_len > lab.numel():
        raise SupervisedBoundaryError(f"qid={question_id}: prompt_len exceeds labels")
    if torch.any(lab[:prompt_len] != -100):
        raise SupervisedBoundaryError(
            f"qid={question_id}: prompt region contains supervised tokens"
        )
    supervised = lab[prompt_len:]
    if supervised.numel() == 0:
        raise SupervisedBoundaryError(f"qid={question_id}: zero supervised tokens")
    if torch.any(supervised == -100):
        raise SupervisedBoundaryError(
            f"qid={question_id}: masked tokens inside assistant suffix"
        )
    if int(supervised[-1].item()) != int(eos_id):
        raise SupervisedBoundaryError(
            f"qid={question_id}: last supervised token is not eos "
            f"(got id={int(supervised[-1].item())})"
        )
    # Exactly one eos, and it must be the final supervised token.
    eos_hits = (supervised == int(eos_id)).nonzero(as_tuple=False).view(-1)
    if eos_hits.numel() != 1:
        raise SupervisedBoundaryError(
            f"qid={question_id}: expected exactly one supervised eos, "
            f"found {int(eos_hits.numel())} at relative positions "
            f"{eos_hits.tolist()}"
        )
    if int(eos_hits[0].item()) != supervised.numel() - 1:
        raise SupervisedBoundaryError(
            f"qid={question_id}: supervised eos is not terminal"
        )
    if supervised.numel() < 2:
        raise SupervisedBoundaryError(
            f"qid={question_id}: supervised region must include answer + eos"
        )


def build_supervised_example(
    loaded: LoadedInternVL,
    *,
    image: Image.Image,
    question: str,
    target_answer: str,
    instruction: str = DEFAULT_INSTRUCTION,
    max_seq_length: int = 8192,
    question_id: int = -1,
) -> SupervisedExample:
    """Build teacher-forced tensors; supervise assistant answer + one end-of-turn.

    Token boundaries are established by tokenizing the prompt-only conversation
    and the full conversation as complete strings (no string-concat of token ids).
    """
    model = loaded.model
    tokenizer = loaded.tokenizer
    img_ctx_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_ctx_id

    pixel_values, _tile = load_pixel_values(
        image,
        input_size=loaded.image_size,
        max_num=loaded.max_num,
        use_thumbnail=loaded.use_thumbnail,
    )
    num_patches = int(pixel_values.shape[0])
    expected_img_tokens = int(num_patches * loaded.num_image_token_per_tile)

    text = f"{question.strip()}\n{instruction}"
    question_marked = f"<image>\n{text}"
    answer = str(target_answer).strip()
    if not answer:
        raise ValueError(f"empty target_answer for qid={question_id}")

    template = internvl_conv.get_conv_template(
        loaded.template_name or model.template, model=model
    )
    template.system_message = model.system_message
    # sep is typically "<|im_end|>\n"; the end *token* is the stripped form.
    eos_str = template.sep.strip()
    if not eos_str:
        raise SupervisedBoundaryError("conversation template sep is empty after strip")
    eos_id = int(tokenizer.convert_tokens_to_ids(eos_str))
    if eos_id < 0:
        raise SupervisedBoundaryError(f"eos token {eos_str!r} missing from tokenizer")

    # Prompt-only (assistant turn open, no answer) — same as inference path.
    t_prompt = internvl_conv.get_conv_template(
        loaded.template_name or model.template, model=model
    )
    t_prompt.system_message = model.system_message
    t_prompt.append_message(t_prompt.roles[0], question_marked)
    t_prompt.append_message(t_prompt.roles[1], None)
    prompt_query = _expand_image_tokens(
        t_prompt.get_prompt(),
        num_patches=num_patches,
        num_image_token=loaded.num_image_token_per_tile,
    )

    # Full conversation with answer text in assistant turn.
    t_full = internvl_conv.get_conv_template(
        loaded.template_name or model.template, model=model
    )
    t_full.system_message = model.system_message
    t_full.append_message(t_full.roles[0], question_marked)
    t_full.append_message(t_full.roles[1], answer)
    full_query = _expand_image_tokens(
        t_full.get_prompt(),
        num_patches=num_patches,
        num_image_token=loaded.num_image_token_per_tile,
    )

    prompt_ids = tokenizer(prompt_query, return_tensors="pt")["input_ids"][0]
    full_ids = tokenizer(full_query, return_tensors="pt")["input_ids"][0]

    # Verify prompt is an exact prefix of full tokenization.
    prompt_len = int(prompt_ids.numel())
    if full_ids.numel() < prompt_len or not torch.equal(
        full_ids[:prompt_len], prompt_ids
    ):
        max_match = 0
        upto = min(prompt_len, int(full_ids.numel()))
        for i in range(upto):
            if int(full_ids[i]) != int(prompt_ids[i]):
                break
            max_match = i + 1
        raise ValueError(
            f"qid={question_id}: prompt tokens are not a prefix of full tokens "
            f"(matched {max_match}/{prompt_len}; full_len={full_ids.numel()}). "
            "Refusing assumed concatenation."
        )

    full_ids = resolve_assistant_termination(
        full_ids, prompt_len=prompt_len, eos_id=eos_id
    )

    seq_len = int(full_ids.numel())
    if seq_len > int(max_seq_length):
        raise ValueError(
            f"qid={question_id}: sequence length {seq_len} exceeds "
            f"max_seq_length={max_seq_length} (no silent truncation)"
        )

    n_img = int((full_ids == img_ctx_id).sum().item())
    if n_img != expected_img_tokens:
        raise ValueError(
            f"qid={question_id}: IMG_CONTEXT count {n_img} != "
            f"expected {expected_img_tokens} (patches={num_patches})"
        )

    labels = full_ids.clone()
    labels[:prompt_len] = -100
    assert_supervised_termination(
        labels, prompt_len=prompt_len, eos_id=eos_id, question_id=question_id
    )
    n_sup = int((labels != -100).sum().item())

    attention_mask = torch.ones_like(full_ids)
    image_flags = torch.ones(num_patches, dtype=torch.long)

    return SupervisedExample(
        question_id=int(question_id),
        question=question,
        target_answer=answer,
        input_ids=full_ids.unsqueeze(0),
        attention_mask=attention_mask.unsqueeze(0),
        labels=labels.unsqueeze(0),
        pixel_values=pixel_values,
        image_flags=image_flags,
        num_patches=num_patches,
        seq_len=seq_len,
        n_supervised_tokens=n_sup,
        n_img_context=n_img,
        prompt_len=prompt_len,
        assistant_eos_offset=seq_len - 1,
    )


def label_mask_excerpt(labels: torch.Tensor, tokenizer, *, max_show: int = 32) -> str:
    """Readable excerpt of supervised tokens (skip long image regions)."""
    ids = labels.view(-1).tolist()
    parts = []
    i = 0
    while i < len(ids) and len(parts) < max_show:
        if ids[i] == -100:
            j = i
            while j < len(ids) and ids[j] == -100:
                j += 1
            parts.append(f"[MASK×{j - i}]")
            i = j
            continue
        tok = tokenizer.convert_ids_to_tokens(int(ids[i]))
        parts.append(tok)
        i += 1
    return " ".join(parts)


def collate_supervised(examples: list[SupervisedExample], *, pad_token_id: int) -> dict[str, Any]:
    """Microbatch collate (expected microbatch size 1 in Phase 3A/3C)."""
    if len(examples) != 1:
        raise ValueError("Phase 3A/3C collate supports microbatch size 1 only")
    ex = examples[0]
    return {
        "input_ids": ex.input_ids,
        "attention_mask": ex.attention_mask,
        "labels": ex.labels,
        "pixel_values": ex.pixel_values,
        "image_flags": ex.image_flags.unsqueeze(-1),
        "question_id": ex.question_id,
        "num_patches": ex.num_patches,
        "seq_len": ex.seq_len,
        "n_supervised_tokens": ex.n_supervised_tokens,
    }
