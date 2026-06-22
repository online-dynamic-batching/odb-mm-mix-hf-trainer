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
  -> odb_mm_mix.MMMixDataset
  -> transformers Trainer
  -> odb.apply(...)
  -> odb.integrations.hf.configure_trainer(...)
```

## Prepare Data

```bash
git clone https://github.com/online-dynamic-batching/odb-mm-mix-example.git
cd odb-mm-mix-example
pip install -e .
python scripts/build_public_mm_mix.py --output ../data/mm-mix-tmdb --overwrite
```

## Run

```bash
python scripts/train_hf_trainer_proxy.py \
  --data data/mm-mix-tmdb \
  --token-budget 8192 \
  --max-steps 100
```

The script trains a tiny proxy language model over MM-Mix record lengths. This
is an integration and batching-behavior example, not a paper reproduction.

For real model training, keep your existing tokenizer, processor, collator, and
model forward path. The key ODB changes are the ODB-enabled DataLoader and the
Trainer metadata/loss-scaling hook.

