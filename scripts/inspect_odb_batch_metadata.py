#!/usr/bin/env python3
"""Inspect ODB batch metadata for the direct HF MM-Mix path."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import odb
from odb.constants import (
    LOCAL_BATCH_SIZE_KEY,
    LOCAL_TOKENS_KEY,
    TOTAL_BATCH_SIZE_KEY,
    TOTAL_TOKENS_KEY,
)
from odb_mm_mix import DirectReadMMMixDataset
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoProcessor

from hf_mm_utils import make_model_collator
from train_hf_trainer_real_processor import count_records


def _scalar(value: Any) -> int | float | str | None:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return value.reshape(-1)[0].item()
    if isinstance(value, (list, tuple)):
        return _scalar(value[0]) if value else None
    if isinstance(value, (int, float, str)):
        return value
    return None


def _tensor_shape(value: Any) -> list[int] | None:
    return list(value.shape) if isinstance(value, torch.Tensor) else None


def _patch_count(batch: dict[str, Any]) -> int:
    for key in ("pixel_values", "pixel_values_videos"):
        value = batch.get(key)
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.shape[0])
    return 0


def _label_stats(batch: dict[str, Any]) -> dict[str, int | None]:
    labels = batch.get("labels")
    if not isinstance(labels, torch.Tensor):
        return {
            "label_tokens": None,
            "label_min": None,
            "label_max": None,
        }
    valid = labels[labels != -100]
    if valid.numel() == 0:
        return {
            "label_tokens": 0,
            "label_min": None,
            "label_max": None,
        }
    return {
        "label_tokens": int(valid.numel()),
        "label_min": int(valid.min().item()),
        "label_max": int(valid.max().item()),
    }


def _loss_scale_from_metadata(batch: dict[str, Any], world_size: int) -> float | None:
    local_tokens = _scalar(batch.get(LOCAL_TOKENS_KEY))
    total_tokens = _scalar(batch.get(TOTAL_TOKENS_KEY))
    if not isinstance(local_tokens, (int, float)) or not isinstance(
        total_tokens, (int, float)
    ):
        return None
    if local_tokens <= 0 or total_tokens <= 0:
        return None
    return float(local_tokens) / float(total_tokens) * float(world_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", default=os.getenv("ODB_MM_MIX_DATA", "data/mm-mix-tmdb")
    )
    parser.add_argument(
        "--model", default=os.getenv("ODB_MM_MIX_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=int(os.getenv("ODB_MM_MIX_TOKEN_BUDGET", "12288")),
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=int(os.getenv("ODB_MM_MIX_BUFFER_SIZE", "1024")),
    )
    parser.add_argument(
        "--max-patches", type=int, default=int(os.getenv("ODB_MM_MIX_MAX_PATCHES", "0"))
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=int(os.getenv("ODB_MM_MIX_MAX_LENGTH", "2048")),
    )
    parser.add_argument(
        "--image-max-pixels",
        type=int,
        default=int(os.getenv("ODB_MM_MIX_IMAGE_MAX_PIXELS", "589824")),
    )
    parser.add_argument(
        "--processor-backend",
        default=os.getenv("ODB_MM_MIX_PROCESSOR_BACKEND", "auto"),
    )
    parser.add_argument(
        "--num-workers", type=int, default=int(os.getenv("ODB_MM_MIX_NUM_WORKERS", "4"))
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=int(os.getenv("ODB_MM_MIX_PREFETCH_FACTOR", "128")),
    )
    parser.add_argument(
        "--steps", type=int, default=int(os.getenv("ODB_MM_MIX_INSPECT_STEPS", "12"))
    )
    parser.add_argument(
        "--seed", type=int, default=int(os.getenv("ODB_MM_MIX_SEED", "42"))
    )
    parser.add_argument(
        "--loss-scaling", default=os.getenv("ODB_MM_MIX_LOSS_SCALING", "exact")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.multiprocessing.set_sharing_strategy("file_system")
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    data_path = Path(args.data)
    if count_records(data_path) <= 0:
        raise SystemExit(f"No records found in {data_path}")

    processor = AutoProcessor.from_pretrained(
        args.model, trust_remote_code=True, use_fast=True
    )
    dataset = DirectReadMMMixDataset(
        data_path,
        processor=processor,
        max_length=args.max_length,
        image_max_pixels=args.image_max_pixels if args.image_max_pixels > 0 else None,
        processor_backend=args.processor_backend,
    )
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=False,
        )
        if world_size > 1
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=make_model_collator(processor),
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        pin_memory=False,
        sampler=sampler,
        shuffle=sampler is None,
    )
    odb.apply(
        loader,
        token_budget=args.token_budget,
        buffer_size=args.buffer_size,
        max_patches=args.max_patches,
        loss_scaling=args.loss_scaling,
        join=True,
    )

    if rank == 0:
        print(
            json.dumps(
                {
                    "event": "config",
                    "records": len(dataset),
                    "world_size": world_size,
                    "token_budget": args.token_budget,
                    "buffer_size": args.buffer_size,
                    "max_patches": args.max_patches,
                    "max_length": args.max_length,
                    "image_max_pixels": args.image_max_pixels,
                    "processor_backend": args.processor_backend,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    for step, batch in enumerate(loader, start=1):
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        local_row = {
            "rank": rank,
            "step": step,
            "batch_shape": _tensor_shape(input_ids),
            "attention_tokens": int(attention_mask.sum().item())
            if isinstance(attention_mask, torch.Tensor)
            else None,
            "local_batch_size": _scalar(batch.get(LOCAL_BATCH_SIZE_KEY)),
            "all_samples_this_step": _scalar(batch.get(TOTAL_BATCH_SIZE_KEY)),
            "local_tokens": _scalar(batch.get(LOCAL_TOKENS_KEY)),
            "total_tokens": _scalar(batch.get(TOTAL_TOKENS_KEY)),
            "loss_scale": _loss_scale_from_metadata(batch, world_size),
            "patches": _patch_count(batch),
        }
        local_row.update(_label_stats(batch))
        rows = [None for _ in range(world_size)] if rank == 0 else None
        if world_size > 1:
            dist.gather_object(local_row, object_gather_list=rows, dst=0)
        else:
            rows = [local_row]
        if rank == 0:
            print(
                json.dumps(
                    {"event": "batch", "step": step, "ranks": rows}, sort_keys=True
                ),
                flush=True,
            )
        if step >= args.steps:
            break

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
