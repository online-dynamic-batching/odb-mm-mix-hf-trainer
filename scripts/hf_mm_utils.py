"""Small helpers shared by the HF Trainer MM-Mix examples."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from odb_mm_mix import collate_tokens
import torch


VISION_SPECIAL_TOKENS = (
    "<image>",
    "<|image_pad|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|video_pad|>",
)


def collect_vision_token_ids(processor: Any) -> set[int]:
    """Collect common vision special-token ids for label masking checks."""
    ids: set[int] = set()
    value = getattr(processor, "image_token_id", None)
    if value is not None:
        ids.add(int(value))
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        for token in VISION_SPECIAL_TOKENS:
            try:
                token_id = tokenizer.convert_tokens_to_ids(token)
            except Exception:
                continue
            if (
                isinstance(token_id, int)
                and token_id >= 0
                and token_id != getattr(tokenizer, "unk_token_id", None)
            ):
                ids.add(token_id)
    return ids


def mask_vision_tokens(
    batch: dict[str, Any], vision_token_ids: set[int]
) -> dict[str, Any]:
    """Mask known vision special tokens in labels if they are present."""
    input_ids = batch.get("input_ids")
    labels = batch.get("labels")
    if not isinstance(input_ids, torch.Tensor) or not isinstance(labels, torch.Tensor):
        return batch
    for token_id in vision_token_ids:
        labels[input_ids == int(token_id)] = -100
    return batch


def make_model_collator(
    processor: Any,
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Return a collator whose output can be passed directly to HF VLM models."""
    vision_token_ids = collect_vision_token_ids(processor)

    def _collate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        batch = collate_tokens(rows)
        batch.pop("odb_n_patches", None)
        return mask_vision_tokens(batch, vision_token_ids)

    return _collate


def tensor_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, torch.Tensor):
        return None
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": int(value.numel()),
    }
