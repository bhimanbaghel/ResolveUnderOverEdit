# [Resolving UnderEdit & OverEdit with Iterative & Neighbor-Assisted Model Editing](https://arxiv.org/abs/2503.11895)

**Official implementation** (Accepted at EMNLP 2025 Findings)

This repository provides an iterative editing interface for large language models with a simple CLI to reproduce experiments and run your own edits.

## Setup

### Prerequisites

This project uses `uv` for dependency management. If you don't have `uv` installed, please follow the installation instructions at:
- **Official uv documentation**: https://docs.astral.sh/uv/getting-started/installation/

### Installation

1. Clone the repository:
```bash
git clone https://github.com/bhimanbaghel/ResolveUnderOverEdit.git
cd ResolveUnderOverEdit
```

2. Install dependencies using `uv`:
```bash
uv sync
```

This command will:
- Create a virtual environment (`.venv`)
- Install all dependencies with exact versions from `uv.lock`

3. Run the code:
```bash
# Option 1: Activate the virtual environment
source .venv/bin/activate
python checkZ.py --help

# Option 2: Run directly with uv (no activation needed)
uv run python checkZ.py --help
```

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

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@misc{baghel2025resolvingundereditoveredit,
      title={Resolving UnderEdit & OverEdit with Iterative & Neighbor-Assisted Model Editing}, 
      author={Bhiman Kumar Baghel and Emma Jordan and Zheyuan Ryan Shi and Xiang Lorraine Li},
      year={2025},
      eprint={2503.11895},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.11895}, 
}
```

## Acknowledgments

This work builds upon the [EasyEdit](https://github.com/zjunlp/EasyEdit) framework. We extend our sincere gratitude to the EasyEdit authors for their excellent work and open-source contribution. If you use this code, please also consider citing their work:

```bibtex
@article{wang2023easyedit,
  title={Easyedit: An easy-to-use knowledge editing framework for large language models},
  author={Wang, Peng and Zhang, Ningyu and Xie, Xin and Yao, Yunzhi and Tian, Bozhong and Wang, Mengru and Xi, Zekun and Cheng, Siyuan and Liu, Kangwei and Zheng, Guozhou and others},
  journal={arXiv preprint arXiv:2308.07269},
  year={2023}
}
```
