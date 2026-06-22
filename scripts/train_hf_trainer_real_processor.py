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
    parser.add_argument("--data", default="data/mm-mix-tmdb")
    parser.add_argument("--source-data", default=os.getenv("ODB_MM_MIX_SOURCE_DATA"))
    parser.add_argument("--local-data", default=os.getenv("ODB_MM_MIX_LOCAL_DATA"))
    parser.add_argument("--force-local-copy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--model", default=os.getenv("ODB_MM_MIX_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"))
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loader", choices=["odb", "standard"], default=os.getenv("ODB_MM_MIX_LOADER", "odb"))
    parser.add_argument("--output-dir", default=os.getenv("ODB_MM_MIX_OUTPUT_DIR", "outputs/hf-trainer-real"))
    parser.add_argument("--token-budget", type=int, default=int(os.getenv("ODB_MM_MIX_TOKEN_BUDGET", "8192")))
    parser.add_argument("--buffer-size", type=int, default=int(os.getenv("ODB_MM_MIX_BUFFER_SIZE", "512")))
    parser.add_argument("--fixed-batch-size", type=int, default=int(os.getenv("ODB_MM_MIX_FIXED_BATCH_SIZE", "1")))
    parser.add_argument("--max-length", type=int, default=int(os.getenv("ODB_MM_MIX_MAX_LENGTH", "4096")))
    parser.add_argument("--max-steps", type=int, default=int(os.getenv("ODB_MM_MIX_MAX_STEPS", "20")))
    parser.add_argument("--num-train-epochs", type=float, default=float(os.getenv("ODB_MM_MIX_EPOCHS", "1.0")))
    parser.add_argument("--num-workers", type=int, default=int(os.getenv("ODB_MM_MIX_NUM_WORKERS", "2")))
    parser.add_argument("--prefetch-factor", type=int, default=int(os.getenv("ODB_MM_MIX_PREFETCH_FACTOR", "64")))
    parser.add_argument("--lr", type=float, default=float(os.getenv("ODB_MM_MIX_LR", "1e-5")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("ODB_MM_MIX_SEED", "42")))
    parser.add_argument("--join", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loss-scaling", default=os.getenv("ODB_MM_MIX_LOSS_SCALING", "exact"))
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=torch.cuda.is_available())
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-strategy", default="no")
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
    }
    return TrainingArguments(**{k: v for k, v in common.items() if v is not None})


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.multiprocessing.set_sharing_strategy("file_system")

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
    model = load_model(args.model, trust_remote_code=args.trust_remote_code, dtype=dtype)
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass
    trainable_keywords = tuple(x.strip() for x in args.trainable_keywords.split(",") if x.strip())
    trainable = configure_trainable_parameters(model, trainable_keywords)
    if trainable <= 0:
        raise SystemExit(f"No trainable parameters matched: {trainable_keywords}")

    dataset = DirectReadMMMixDataset(data_path, processor=processor, max_length=args.max_length)
    training_args = make_training_args(args)
    trainer = ODBTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=make_model_collator(processor),
    )

    if args.loader == "odb":
        train_loader = trainer.get_train_dataloader()
        handle = odb.apply(
            train_loader,
            token_budget=args.token_budget,
            buffer_size=args.buffer_size,
            loss_scaling=args.loss_scaling,
            join=args.join,
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
                "records": len(dataset),
                "model": args.model,
                "trainable_parameters": trainable,
                "token_budget": args.token_budget if args.loader == "odb" else None,
                "fixed_batch_size": args.fixed_batch_size if args.loader == "standard" else None,
                "max_length": args.max_length,
                "max_steps": args.max_steps,
            },
            indent=2,
        ),
        flush=True,
    )
    trainer.train()
    metrics_path = Path(args.output_dir) / f"train_metrics_{args.loader}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(trainer.state.log_history, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
