#!/usr/bin/env python3
"""HF Trainer real-processor MM-Mix example using the ODB pip package."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import odb
from odb.integrations.hf import ODBTrainer, configure_trainer
from odb_mm_mix import DirectReadMMMixDataset
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoProcessor, TrainingArguments

from hf_mm_utils import make_model_collator


def count_records(path: Path) -> int:
    metadata = path / "metadata.json"
    if metadata.exists():
        try:
            return int(json.loads(metadata.read_text(encoding="utf-8")).get("num_records") or 0)
        except Exception:
            pass
    records = path / "records.jsonl"
    if not records.exists():
        return 0
    with records.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def copy_tree_if_needed(source: Path, target: Path, *, force: bool = False) -> Path:
    if count_records(source) <= 0:
        raise SystemExit(f"source TMDB is missing or empty: {source}")
    if force and target.exists():
        shutil.rmtree(target)
    if count_records(target) == count_records(source):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    rsync = shutil.which("rsync")
    if rsync:
        target.mkdir(parents=True, exist_ok=True)
        import subprocess

        subprocess.check_call([rsync, "-a", "--delete", f"{source}/", f"{target}/"])
    else:
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
    return target


def load_model(model_name_or_path: str, *, trust_remote_code: bool, dtype: torch.dtype):
    import transformers

    model_cls = getattr(transformers, "AutoModelForImageTextToText", None)
    if model_cls is None:
        model_cls = getattr(transformers, "AutoModelForVision2Seq")
    try:
        return model_cls.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
    except ValueError:
        from transformers import AutoModelForVision2Seq

        return AutoModelForVision2Seq.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )


def configure_processor_pixels(processor: Any, *, image_max_pixels: int | None) -> None:
    if image_max_pixels is None or image_max_pixels <= 0:
        return
    targets = [processor, getattr(processor, "image_processor", None)]
    for target in targets:
        if target is None:
            continue
        for name in ("max_pixels", "image_max_pixels"):
            if hasattr(target, name):
                try:
                    setattr(target, name, int(image_max_pixels))
                except Exception:
                    pass


def configure_trainable_parameters(model: torch.nn.Module, trainable_keywords: tuple[str, ...]) -> int:
    if any(keyword.lower() in {"*", "all", "full"} for keyword in trainable_keywords):
        for param in model.parameters():
            param.requires_grad_(True)
        return sum(param.numel() for param in model.parameters())

    trainable = 0
    for name, param in model.named_parameters():
        keep = any(keyword in name for keyword in trainable_keywords)
        param.requires_grad_(keep)
        if keep:
            trainable += param.numel()
    return trainable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.getenv("ODB_MM_MIX_DATA", "data/mm-mix-tmdb"))
    parser.add_argument("--source-data", default=os.getenv("ODB_MM_MIX_SOURCE_DATA"))
    parser.add_argument("--local-data", default=os.getenv("ODB_MM_MIX_LOCAL_DATA"))
    parser.add_argument("--force-local-copy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--model", default=os.getenv("ODB_MM_MIX_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"))
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loader", choices=["odb", "standard"], default=os.getenv("ODB_MM_MIX_LOADER", "odb"))
    parser.add_argument("--output-dir", default=os.getenv("ODB_MM_MIX_OUTPUT_DIR", "outputs/hf-trainer-real"))
    parser.add_argument("--token-budget", type=int, default=int(os.getenv("ODB_MM_MIX_TOKEN_BUDGET", "12288")))
    parser.add_argument("--buffer-size", type=int, default=int(os.getenv("ODB_MM_MIX_BUFFER_SIZE", "1024")))
    parser.add_argument("--max-patches", type=int, default=int(os.getenv("ODB_MM_MIX_MAX_PATCHES", "0")))
    parser.add_argument("--fixed-batch-size", type=int, default=int(os.getenv("ODB_MM_MIX_FIXED_BATCH_SIZE", "1")))
    parser.add_argument("--max-length", type=int, default=int(os.getenv("ODB_MM_MIX_MAX_LENGTH", "2048")))
    parser.add_argument("--train-size", type=int, default=int(os.getenv("ODB_MM_MIX_TRAIN_SIZE", "0")))
    parser.add_argument("--image-max-pixels", type=int, default=int(os.getenv("ODB_MM_MIX_IMAGE_MAX_PIXELS", "0")))
    parser.add_argument("--max-steps", type=int, default=int(os.getenv("ODB_MM_MIX_MAX_STEPS", "20")))
    parser.add_argument("--num-train-epochs", type=float, default=float(os.getenv("ODB_MM_MIX_EPOCHS", "1.0")))
    parser.add_argument("--num-workers", type=int, default=int(os.getenv("ODB_MM_MIX_NUM_WORKERS", "4")))
    parser.add_argument("--prefetch-factor", type=int, default=int(os.getenv("ODB_MM_MIX_PREFETCH_FACTOR", "128")))
    parser.add_argument("--lr", type=float, default=float(os.getenv("ODB_MM_MIX_LR", "1e-5")))
    parser.add_argument("--lr-scheduler-type", default=os.getenv("ODB_MM_MIX_LR_SCHEDULER_TYPE", "cosine"))
    parser.add_argument("--warmup-ratio", type=float, default=float(os.getenv("ODB_MM_MIX_WARMUP_RATIO", "0.03")))
    parser.add_argument("--max-grad-norm", type=float, default=float(os.getenv("ODB_MM_MIX_MAX_GRAD_NORM", "4.0")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("ODB_MM_MIX_SEED", "42")))
    parser.add_argument("--deepspeed", default=os.getenv("ODB_MM_MIX_DEEPSPEED"))
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("ODB_MM_MIX_GRADIENT_CHECKPOINTING", "1").lower() in {"1", "true", "yes", "y"},
    )
    parser.add_argument("--join", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loss-scaling", default=os.getenv("ODB_MM_MIX_LOSS_SCALING", "exact"))
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=torch.cuda.is_available())
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-strategy", default=os.getenv("ODB_MM_MIX_SAVE_STRATEGY", "no"))
    parser.add_argument(
        "--save-final-model",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("ODB_MM_MIX_SAVE_FINAL_MODEL", "0").lower() in {"1", "true", "yes", "y"},
    )
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument(
        "--trainable-keywords",
        default=os.getenv("ODB_MM_MIX_TRAINABLE_KEYWORDS", "norm,merger"),
        help="Comma-separated parameter-name fragments to train; use 'full' for full fine-tuning.",
    )
    return parser.parse_args()


def make_training_args(args: argparse.Namespace) -> TrainingArguments:
    common: dict[str, Any] = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": 1 if args.loader == "odb" else args.fixed_batch_size,
        "dataloader_num_workers": args.num_workers,
        "dataloader_prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
        "learning_rate": args.lr,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "save_strategy": args.save_strategy,
        "report_to": [],
        "remove_unused_columns": False,
        "logging_steps": args.logging_steps,
        "seed": args.seed,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "deepspeed": args.deepspeed,
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    return TrainingArguments(**{k: v for k, v in common.items() if v is not None})


def maybe_limit_train_dataset(dataset, train_size: int):
    if train_size <= 0:
        return dataset
    if train_size > len(dataset):
        raise SystemExit(f"train_size={train_size} exceeds dataset size {len(dataset)}")
    return Subset(dataset, range(train_size))


def make_odb_train_dataloader(args: argparse.Namespace, dataset, collator) -> DataLoader:
    sampler = None
    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
            seed=args.seed,
            drop_last=False,
        )
    kwargs: dict[str, Any] = {
        "batch_size": 1,
        "collate_fn": collator,
        "num_workers": args.num_workers,
        "pin_memory": False,
        "sampler": sampler,
        "shuffle": sampler is None,
    }
    if args.num_workers > 0:
        kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(dataset, **kwargs)


def validate_single_process_device_context() -> None:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if world_size > 1 or not torch.cuda.is_available():
        return
    if torch.cuda.device_count() > 1:
        raise SystemExit(
            "This single-process HF Trainer example expects one visible GPU. "
            "Set CUDA_VISIBLE_DEVICES to one device or launch with a distributed runner."
        )


def last_train_metrics(log_history: list[dict[str, Any]]) -> dict[str, Any]:
    for row in reversed(log_history):
        if "train_runtime" in row:
            return row
    return {}


def write_training_outputs(args: argparse.Namespace, trainer: ODBTrainer, train_metrics: dict[str, Any]) -> None:
    if not trainer.is_world_process_zero():
        return
    metrics_path = Path(args.output_dir) / f"train_metrics_{args.loader}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    log_history = trainer.state.log_history
    metrics_path.write_text(json.dumps(log_history, indent=2) + "\n", encoding="utf-8")

    final_metrics = dict(train_metrics)
    final_metrics.update(last_train_metrics(log_history))
    global_step = int(getattr(trainer.state, "global_step", 0) or 0)
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    if args.loader == "odb":
        emitted_samples = int(getattr(trainer.state, "total_data_step", 0) or 0)
    else:
        emitted_samples = global_step * int(args.fixed_batch_size) * int(world_size)
    train_runtime = float(final_metrics.get("train_runtime") or 0.0)
    summary = {
        "loader": args.loader,
        "global_step": global_step,
        "emitted_samples": emitted_samples,
        "mean_emitted_samples_per_step": emitted_samples / global_step if global_step else None,
        "effective_emitted_samples_per_second": emitted_samples / train_runtime if train_runtime > 0 else None,
        "trainer_metrics": final_metrics,
        "world_size": int(world_size),
    }
    summary_path = Path(args.output_dir) / f"train_summary_{args.loader}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.multiprocessing.set_sharing_strategy("file_system")
    validate_single_process_device_context()

    data_path = Path(args.data)
    if args.source_data or args.local_data:
        data_path = copy_tree_if_needed(
            Path(args.source_data or args.data),
            Path(args.local_data or args.data),
            force=args.force_local_copy,
        )
    if count_records(data_path) <= 0:
        raise SystemExit(f"No records found in {data_path}")

    dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=args.trust_remote_code, use_fast=True)
    configure_processor_pixels(processor, image_max_pixels=args.image_max_pixels)
    model = load_model(args.model, trust_remote_code=args.trust_remote_code, dtype=dtype)
    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
    trainable_keywords = tuple(x.strip() for x in args.trainable_keywords.split(",") if x.strip())
    trainable = configure_trainable_parameters(model, trainable_keywords)
    if trainable <= 0:
        raise SystemExit(f"No trainable parameters matched: {trainable_keywords}")

    raw_dataset = DirectReadMMMixDataset(data_path, processor=processor, max_length=args.max_length)
    dataset = maybe_limit_train_dataset(raw_dataset, args.train_size)
    training_args = make_training_args(args)
    collator = make_model_collator(processor)
    trainer = ODBTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    if args.loader == "odb":
        train_loader = make_odb_train_dataloader(args, dataset, collator)
        handle = odb.apply(
            train_loader,
            token_budget=args.token_budget,
            buffer_size=args.buffer_size,
            loss_scaling=args.loss_scaling,
            join=args.join,
            max_patches=args.max_patches,
        )
        configure_trainer(
            trainer,
            dataloader=train_loader,
            handle=handle,
            sample_budget=len(dataset),
            max_optimizer_steps=args.max_steps if args.max_steps > 0 else None,
            max_steps_policy="overwrite",
            scheduler_progress="samples",
        )

    print(
        json.dumps(
            {
                "loader": args.loader,
                "data": str(data_path),
                "raw_records": len(raw_dataset),
                "records": len(dataset),
                "model": args.model,
                "trainable_parameters": trainable,
                "token_budget": args.token_budget if args.loader == "odb" else None,
                "max_patches": args.max_patches if args.loader == "odb" else None,
                "fixed_batch_size": args.fixed_batch_size if args.loader == "standard" else None,
                "max_length": args.max_length,
                "image_max_pixels": args.image_max_pixels,
                "deepspeed": args.deepspeed,
                "gradient_checkpointing": args.gradient_checkpointing,
                "max_steps": args.max_steps,
            },
            indent=2,
        ),
        flush=True,
    )
    train_output = trainer.train()
    write_training_outputs(args, trainer, getattr(train_output, "metrics", {}))
    if args.save_final_model and trainer.is_world_process_zero():
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
