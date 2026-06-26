#!/usr/bin/env python3
"""Inspect HF processor multimodal token handling on public MM-Mix records."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from odb_mm_mix import DirectReadMMMixDataset
import torch
from transformers import AutoProcessor

from hf_mm_utils import collect_vision_token_ids, make_model_collator, tensor_summary


VISION_KEYS = ("pixel_values", "image_grid_thw", "image_sizes", "num_items_in_batch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", default=os.getenv("ODB_MM_MIX_DATA", "data/mm-mix-tmdb")
    )
    parser.add_argument(
        "--model", default=os.getenv("ODB_MM_MIX_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=int(os.getenv("ODB_MM_MIX_MAX_LENGTH", "16384")),
    )
    parser.add_argument(
        "--image-max-pixels",
        type=int,
        default=int(os.getenv("ODB_MM_MIX_IMAGE_MAX_PIXELS", "9437184")),
    )
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--output", default=os.getenv("ODB_MM_MIX_INSPECT_OUTPUT"))
    parser.add_argument(
        "--processor-backend", default=os.getenv("ODB_MM_MIX_PROCESSOR_BACKEND", "auto")
    )
    parser.add_argument(
        "--trust-remote-code", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def has_image_record(record: dict[str, Any]) -> bool:
    return bool(record.get("images"))


def raw_record_at(dataset: DirectReadMMMixDataset, index: int) -> dict[str, Any]:
    if hasattr(dataset, "raw_record_at"):
        return dataset.raw_record_at(index)
    return dataset._record_at(index)


def find_indices(dataset: DirectReadMMMixDataset, count: int) -> list[int]:
    image_indices: list[int] = []
    text_indices: list[int] = []
    for index in range(len(dataset)):
        record = raw_record_at(dataset, index)
        if has_image_record(record):
            image_indices.append(index)
        else:
            text_indices.append(index)
        if len(image_indices) >= count and len(text_indices) >= max(1, count // 4):
            break
    return (image_indices[:count] + text_indices[: max(1, count // 4)])[
        : count + max(1, count // 4)
    ]


def inspect_one(
    dataset: DirectReadMMMixDataset, index: int, vision_token_ids: set[int]
) -> dict[str, Any]:
    record = raw_record_at(dataset, index)
    item = dataset[index]
    collator = make_model_collator(dataset.processor)
    batch = collator([item])
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    vision_positions = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in vision_token_ids:
        vision_positions |= input_ids == int(token_id)
    vision_label_values = (
        labels[vision_positions].unique().detach().cpu().tolist()
        if vision_positions.any()
        else []
    )
    vision_keys = {
        key: tensor_summary(batch.get(key))
        for key in VISION_KEYS
        if tensor_summary(batch.get(key))
    }
    return {
        "index": index,
        "source": record.get("source"),
        "has_image": has_image_record(record),
        "num_record_images": len(record.get("images") or []),
        "input_length": int(input_ids.shape[-1]),
        "label_tokens": int((labels != -100).sum().item()),
        "known_vision_token_count": int(vision_positions.sum().item()),
        "known_vision_label_values": vision_label_values,
        "vision_tensors": vision_keys,
        "model_batch_keys": sorted(batch.keys()),
    }


def main() -> None:
    args = parse_args()
    processor = AutoProcessor.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code, use_fast=True
    )
    dataset = DirectReadMMMixDataset(
        Path(args.data),
        processor=processor,
        max_length=args.max_length,
        image_max_pixels=args.image_max_pixels if args.image_max_pixels > 0 else None,
        processor_backend=args.processor_backend,
    )
    vision_token_ids = collect_vision_token_ids(processor)
    indices = find_indices(dataset, args.num_samples)
    rows = [inspect_one(dataset, index, vision_token_ids) for index in indices]
    image_rows = [row for row in rows if row["has_image"]]
    failures = []
    for row in image_rows:
        if not row["vision_tensors"]:
            failures.append(
                f"image row has no vision tensors: index={row['index']} source={row['source']}"
            )
        if row["known_vision_token_count"] and row["known_vision_label_values"] != [
            -100
        ]:
            failures.append(f"vision token labels are not masked: index={row['index']}")
    payload = {
        "model": args.model,
        "processor_backend": args.processor_backend,
        "max_length": args.max_length,
        "image_max_pixels": args.image_max_pixels,
        "num_records": len(dataset),
        "vision_token_ids": sorted(vision_token_ids),
        "inspected": rows,
        "failures": failures,
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
