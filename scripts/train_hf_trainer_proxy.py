#!/usr/bin/env python3
"""Hugging Face Trainer proxy training example over public MM-Mix records."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import odb
from odb.integrations.hf import ODBTrainer, configure_trainer
from odb_mm_mix import MMMixDataset, collate_tokens
from torch import nn
import torch.nn.functional as F
from transformers import TrainingArguments


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, labels=None, attention_mask=None, **unused):
        logits = self.proj(self.embed(input_ids))
        if labels is None:
            return SimpleNamespace(logits=logits, loss=logits.sum() * 0.0)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )
        return SimpleNamespace(logits=logits, loss=loss)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/mm-mix-tmdb")
    parser.add_argument("--output-dir", default="outputs/hf-trainer-proxy")
    parser.add_argument("--token-budget", type=int, default=8192)
    parser.add_argument("--buffer-size", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--join", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loss-scaling", default="exact")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = MMMixDataset(args.data, max_length=args.max_length, vocab_size=args.vocab_size)
    model = TinyLM(args.vocab_size, args.hidden_size)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        dataloader_num_workers=args.num_workers,
        dataloader_prefetch_factor=args.prefetch_factor,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        logging_steps=10,
    )
    trainer = ODBTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_tokens,
    )

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
        max_optimizer_steps=args.max_steps,
        max_steps_policy="overwrite",
        scheduler_progress="samples",
    )
    trainer.train()


if __name__ == "__main__":
    main()
