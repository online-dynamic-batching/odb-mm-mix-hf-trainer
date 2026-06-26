# ODB HF Trainer Example

Experimental framework-native MM-Mix example for
[Online Dynamic Batching](https://github.com/online-dynamic-batching/online-dynamic-batching)
with `transformers.Trainer`.

This example uses the ODB pip package:

```bash
pip install -r requirements.txt
```

## Start Here

The root entrypoint is `run.sh`. It keeps the common train/eval split and
model-processing settings in one visible place, so you do not need to guess
which file under `scripts/` starts training.

```bash
# Recommended ODB path: calls enable_odb(...)
./run.sh odb-enable

# Advanced ODB path: explicit odb.apply(...) + configure_trainer(...)
./run.sh odb-manual

# Fixed-batch baseline
./run.sh standard

# Evaluate a saved checkpoint
./run.sh eval-valloss
./run.sh benchmark
```

`run.sh` defaults to `Qwen/Qwen3-VL-2B-Instruct`, the public MM-Mix TMDB under
`data/mm-mix-tmdb`, `split_mode=lf_val_size`, `val_size=0.05`, `split_seed=42`,
`image_max_pixels=589824`, gradient checkpointing enabled, full fine-tuning,
Deepspeed ZeRO-2, and a short `ODB_MM_MIX_MAX_STEPS=20` run. Set
`ODB_MM_MIX_MAX_STEPS=-1` for a full pass and `ODB_MM_MIX_NUM_PROCESSES=8` for
an 8-GPU run:

```bash
ODB_MM_MIX_MAX_STEPS=-1 ODB_MM_MIX_NUM_PROCESSES=8 ./run.sh odb-enable
```

The public scripts are intentionally named by role:

| Script | Purpose |
| --- | --- |
| `scripts/train_hf_trainer.py` | Real HF Trainer multimodal training. |
| `scripts/eval_valloss.py` | Validation loss on the same lazy HF-direct processor path. |
| `scripts/eval_benchmark.py` | Built-in MMMU-MC choice-likelihood benchmark evaluator. |
| `scripts/inspect_data_pipeline.py` | Pre-training check for multimodal tensor output and label masking. |

The training script intentionally supports two ODB integration modes:

| Mode | Command | What it demonstrates |
| --- | --- | --- |
| One-call hook | `./run.sh odb-enable` | Recommended `enable_odb(...)` entrypoint for ODB-ready HF Trainer pipelines. |
| Manual bridge | `./run.sh odb-manual` | Lower-level `odb.apply(...)` plus `configure_trainer(...)`, useful when you want explicit control over the DataLoader handle. |

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
python scripts/inspect_data_pipeline.py \
  --data data/mm-mix-tmdb \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --image-max-pixels 589824 \
  --num-samples 16
```

This verifies that image records produce vision tensors such as `pixel_values`
or `image_grid_thw`, and that known vision special tokens are masked out of the
training labels when present. The inspection uses the same lazy TMDB Dataset as
training; it does not pre-load the full record table.

The Qwen-VL native processor path downsizes images above `--image-max-pixels`
before expanding visual placeholders into model tokens. The default is
`589824` pixels (`768 x 768`), which keeps the example practical on common
development GPUs. Set it to `0` only when you have a separate image-size
policy.

## Validation Artifacts

Machine-readable validation records are kept under [results/](results/) for
reproducibility checks. They are intended to help users confirm the example
pipeline, not to serve as cross-framework benchmark claims. The detailed
full-epoch validation record for the default small-image configuration is in
[`results/h20_qwen3vl2b_full_lfsplit_20260625.json`](results/h20_qwen3vl2b_full_lfsplit_20260625.json).

## Training

For strict train/eval accounting, use the same deterministic split policy for
training and validation loss. The LLaMA-Factory-compatible policy is:

```text
split_mode = lf_val_size
val_size = 0.05
split_seed = 42
train = permutation[val_size:]
eval = permutation[:val_size]
```

Train with ODB:

```bash
./run.sh odb-enable
```

The ODB branch uses the high-level HF integration entry point:

```python
from odb.integrations.hf import enable_odb

enable_odb(
    trainer,
    train_dataloader=train_loader,
    train_dataset=dataset,
    token_budget=12288,
    loss_scaling="exact",
    join=True,
)
```

For the explicit lower-level path:

```bash
./run.sh odb-manual
```

That path calls `odb.apply(...)` on the DataLoader and then
`configure_trainer(...)` on the Trainer. It is kept as an advanced integration
mode, not as the default recommendation.

Run Standard with the same inputs:

```bash
./run.sh standard
```

For longer validation runs that will be evaluated afterwards, pass
`--save-final-model` or set `ODB_MM_MIX_SAVE_FINAL_MODEL=1`; the short examples
avoid saving full model weights by default.

For real model training, keep your existing model forward path. The key ODB
change is to make the Dataset return single-sample tensor dicts before
grouping, then call `enable_odb(...)` on the Trainer and DataLoader.

## Evaluate Saved Checkpoints

Validation loss uses the same lazy HF-direct processor path as training. Use
`lf_val_size` only for checkpoints trained with the matching split:

```bash
python scripts/eval_valloss.py \
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

MMMU-MC evaluation is provided by this repository. It loads the public
`MMMU/MMMU` validation split with `datasets`, scores answer letters A-H by
next-token likelihood, and writes `mmmu_mc_likelihood_results.json`,
`predictions.jsonl`, `excluded.jsonl`, and `score_audit.json`.

```bash
ODB_HF_EVAL_CHECKPOINT=outputs/hf-trainer-real/odb-enable \
./run.sh benchmark
```
