# ODB MM-Mix Hugging Face Trainer Example

Framework-native MM-Mix example for
[Online Dynamic Batching](https://github.com/online-dynamic-batching/online-dynamic-batching)
with `transformers.Trainer`.

This example uses the ODB pip package:

```bash
pip install -r requirements.txt
```

It consumes the shared public MM-Mix TMDB recipe from
[odb-mm-mix-example](https://github.com/online-dynamic-batching/odb-mm-mix-example)
and demonstrates the native Trainer hook:

```text
public MM-Mix TMDB
  -> odb_mm_mix.DirectReadMMMixDataset lazy-seeks one record per __getitem__
  -> Hugging Face AutoProcessor / model
  -> transformers Trainer
  -> odb.apply(...)
  -> odb.integrations.hf.configure_trainer(...)
```

This is a framework-native example. It does not import LLaMA-Factory and does
not claim exact agreement with the paper-aligned LLaMA-Factory path. The goal is
that, when using the same public data, model family, cutoff, and training
budget, throughput/loss/eval stay within a reasonable sanity range while ODB
uses the same pip package interface a normal HF Trainer project would use.

## Prepare Data

```bash
git clone https://github.com/online-dynamic-batching/odb-mm-mix-example.git
cd odb-mm-mix-example
pip install -e .
python scripts/build_public_mm_mix.py --output ../data/mm-mix-tmdb --overwrite
```

## Inspect Multimodal Processing First

Before training, check that the framework-native HF processor path is producing
real multimodal tensors and sane labels:

```bash
python scripts/inspect_hf_processor_mm_tokens.py \
  --data data/mm-mix-tmdb \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --num-samples 16
```

This verifies that image records produce vision tensors such as `pixel_values`
or `image_grid_thw`, and that known vision special tokens are masked out of the
training labels when present. The inspection uses the same lazy TMDB Dataset as
training; it does not pre-load the full record table.

## Run Real Processor Training

Real processor path:

```bash
python scripts/train_hf_trainer_real_processor.py \
  --data data/mm-mix-tmdb \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --loader odb \
  --token-budget 8192 \
  --max-steps 20
```

Run Standard with the same inputs:

```bash
python scripts/train_hf_trainer_real_processor.py \
  --data data/mm-mix-tmdb \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --loader standard \
  --fixed-batch-size 1 \
  --max-steps 20
```

Proxy smoke path:

```bash
python scripts/train_hf_trainer_proxy.py \
  --data data/mm-mix-tmdb \
  --token-budget 8192 \
  --max-steps 100
```

The proxy script trains a tiny language model over MM-Mix record lengths. It is
only a quick adapter smoke test; it does not validate multimodal token
processing and should not be used for quality or throughput comparisons.

For real model training, keep your existing tokenizer, processor, collator, and
model forward path. The key ODB changes are the ODB-enabled DataLoader and the
Trainer metadata/loss-scaling hook.
