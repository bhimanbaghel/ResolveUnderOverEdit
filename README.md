# Resolving UnderEdit & OverEdit with Iterative & Neighbor-Assisted Model Editing

Official implementation of “Resolving UnderEdit & OverEdit with Iterative & Neighbor-Assisted Model Editing”.

This repository provides an iterative editing interface for large language models with a simple CLI to reproduce experiments and run your own edits.

## Quickstart

- Install dependencies for your environment, then run iterative editing using the provided CLI.
- Start with the command template below and adjust placeholders for your setup.

### Command template

```bash
python3 checkZ.py \
  --alg_name=<$EDITING_ALGO>_RECURSIVE \
  --model_name=<$MODEL_NAME> \
  --hparams_fname=<$MODEL_CONFIG> \
  --ds_name=<$DATASET> \
  --num_edits=<$BATCH_SIZE> \
  --ds_subset=<$SUBSET_INDEX> \
  --iterations=<$ITERATIONS>
```

- **<$EDITING_ALGO>**: Base editing algorithm name (e.g., `AlphaEdit`); `_RECURSIVE` will be appended automatically.
- **<$MODEL_NAME>**: Model identifier (e.g., `llama-3-8b`).
- **<$MODEL_CONFIG>**: Path to model/algorithm hyperparameters (YAML).
- **<$DATASET>**: Editing dataset (e.g., `mcf`).
- **<$BATCH_SIZE>**: Number of edits per batch.
- **<$SUBSET_INDEX>**: Dataset subset index (integer).
- **<$ITERATIONS>**: Number of recursive editing passes.

### Minimal example

```bash
python3 checkZ.py --alg_name=AlphaEdit_RECURSIVE \
  --model_name=llama-3-8b \
  --hparams_fname=./hparams/AlphaEdit_RECURSIVE/llama3-8b.yaml \
  --ds_name=mcf \
  --num_edits=10 \
  --ds_subset=1 \
  --iterations=5
```

## Notes

- Ensure the YAML in `--hparams_fname` matches the chosen model and algorithm variant.
- Iterative mode applies edits recursively; `--iterations` controls the number of passes.

## Repository

- GitHub: https://github.com/bhimanbaghel/ResolveUnderOverEdit

