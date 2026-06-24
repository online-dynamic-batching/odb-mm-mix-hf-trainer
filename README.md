# ODB MM-Mix Hugging Face Trainer Example

Experimental framework-native MM-Mix example for
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
  -> Hugging Face AutoProcessor runs per sample in the Dataset
  -> ODB-ready tensor sample dicts with input_ids / labels / vision tensors
  -> ODB groups real post-processor lengths
  -> tensor-only collator pads/stacks each ODB group
  -> transformers Trainer runs model.forward / loss / optimizer
```

This is a framework-native workbench. It does not import LLaMA-Factory and does
not claim agreement with the paper-aligned LLaMA-Factory path. Use it to audit
native HF processor behavior, tensor shapes, ODB metadata, and Trainer hook
mechanics before treating any run as a throughput or quality benchmark.

HF Trainer already supports multimodal model training once each batch contains
the tensors expected by `model.forward`. ODB adds one extra contract: the
model-specific tokenizer/processor/vision expansion must run before grouping,
so ODB can see the true post-processing length of each sample. In this example,
`DirectReadMMMixDataset.__getitem__` returns those tensor samples lazily and
declares `odb_ready = True` after this contract is audited. `enable_odb(...)`
rejects pipelines where raw text/images are still processed inside the collator.

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
  --image-max-pixels 589824 \
  --num-samples 16
```

This verifies that image records produce vision tensors such as `pixel_values`
or `image_grid_thw`, and that known vision special tokens are masked out of the
training labels when present. The inspection uses the same lazy TMDB Dataset as
training; it does not pre-load the full record table.

The Qwen-VL native processor path downsizes images above `--image-max-pixels`
before expanding visual placeholders into model tokens. The default is
`589824` pixels (`768 x 768`), which prevents very large source images from
filling the text cutoff with visual tokens. Set it to `0` only when you have a
separate image-size policy.

## Current Validation Status

The ODB/HF Trainer adapter contract is validated with ODB-ready tensor
datasets and a full-epoch public MM-Mix H20 run using
`online-dynamic-batching==0.1.2`. This is a framework-native HF processor path,
not a paper-aligned LLaMA-Factory reproduction, so use the LLaMA-Factory
MM-Mix reference project when you need the paper-style training stack.

Full-epoch validation on one 8-GPU H20 node with Qwen3-VL-2B-Instruct,
`split_mode=lf_val_size`, `val_size=0.05`, and `split_seed=42`:

| Loader | Samples/s | Speedup | Runtime | Steps | Samples/step | Val loss | Token-weighted val loss | MMMU-MC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard | 18.92 | 1.00x | 10367.6s | 24525 | 8.0 | 1.0363 | 1.2107 | 43.41 |
| ODB | 50.53 | 2.67x | 3882.7s | 1278 | 153.5 | 1.0302 | 1.2075 | 48.35 |

Both runs train over the same 196,200 examples and evaluate on the same 10,326
held-out examples. The detailed machine-readable record is in
[`results/h20_qwen3vl2b_full_lfsplit_20260625.json`](results/h20_qwen3vl2b_full_lfsplit_20260625.json).

## Run Real Processor Training

For strict train/eval accounting, use the same deterministic split policy for
training and validation loss. The LLaMA-Factory-compatible policy is:

```text
split_mode = lf_val_size
val_size = 0.05
split_seed = 42
train = permutation[val_size:]
eval = permutation[:val_size]
```

The older prefix mode (`--split-mode prefix --train-size N`) is useful for
quick debugging, but then validation must use a held-out tail range. Do not
evaluate a prefix-trained checkpoint on `lf_val_size` and call it held-out
validation loss, because most `lf_val_size` examples were already seen during
prefix training.

Real processor path with ODB:

```bash
python scripts/train_hf_trainer_real_processor.py \
  --data data/mm-mix-tmdb \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --loader odb \
  --split-mode lf_val_size \
  --val-size 0.05 \
  --split-seed 42 \
  --token-budget 8192 \
  --image-max-pixels 589824 \
  --max-steps 20
```

The ODB branch uses the high-level HF integration entry point:

```python
from odb.integrations.hf import enable_odb

enable_odb(
    trainer,
    train_dataloader=train_loader,
    train_dataset=dataset,
    token_budget=8192,
    loss_scaling="exact",
    join=True,
)
```

Run Standard with the same inputs:

```bash
python scripts/train_hf_trainer_real_processor.py \
  --data data/mm-mix-tmdb \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --loader standard \
  --split-mode lf_val_size \
  --val-size 0.05 \
  --split-seed 42 \
  --fixed-batch-size 1 \
  --image-max-pixels 589824 \
  --max-steps 20
```

For longer validation runs that will be evaluated afterwards, pass
`--save-final-model` or set `ODB_MM_MIX_SAVE_FINAL_MODEL=1`; the short examples
avoid saving full model weights by default.

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

For real model training, keep your existing model forward path. The key ODB
change is to make the Dataset return single-sample tensor dicts before
grouping, then call `enable_odb(...)` on the Trainer and DataLoader.

## Evaluate Saved Checkpoints

Validation loss uses the same lazy HF-direct processor path as training. Use
`lf_val_size` only for checkpoints trained with the matching split:

```bash
python scripts/eval_hf_valloss.py \
  --checkpoint outputs/hf-trainer-real \
  --data data/mm-mix-tmdb \
  --output-dir outputs/hf-trainer-real/eval_out_hf_valloss \
  --split-mode lf_val_size \
  --val-size 0.05 \
  --split-seed 42
```

The output JSON includes `eval_indices_preview`, `label_tokens`,
`label_tokens_per_sample`, and `token_weighted_eval_loss` so you can audit
whether the validation split and label mask are comparable across runs.

MMMU-MC evaluation reuses the paper evaluator from a local LLaMA-Factory
checkout:

```bash
ODB_HF_EVAL_CHECKPOINT=outputs/hf-trainer-real \
ODB_HF_EVAL_SAVE_DIR=outputs/hf-trainer-real/mmmu_mc_likelihood_hf \
python local_validation/run_hf_mmmu_eval_h20.py
```
