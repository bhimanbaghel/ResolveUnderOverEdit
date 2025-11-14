# [Resolving UnderEdit & OverEdit with Iterative & Neighbor-Assisted Model Editing](https://aclanthology.org/2025.findings-emnlp.798/)

[![arXiv](https://img.shields.io/badge/arXiv-2503.11895-b31b1b.svg)](https://arxiv.org/abs/2503.11895)
[![EMNLP 2025](https://img.shields.io/badge/EMNLP-2025-blue.svg)](https://aclanthology.org/2025.findings-emnlp.798/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Official Implementation** — Accepted at EMNLP 2025 Findings

![Model Editing Example](assets/EditExample.png)

This repository provides the implementation of iterative and neighbor-assisted model editing for large language models with a simple CLI to reproduce experiments and run your own edits.

## ⚙️ Setup

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

## 📥 Downloading Model Statistics

The experiments require pre-computed model statistics (~13GB total) that are hosted on Hugging Face Hub.

### Prerequisites

The Hugging Face Hub library is required for downloading. It should already be available if you ran `uv sync`. If needed, you can add it:

```bash
uv add huggingface_hub
```

### Download All Statistics

To download statistics for all models:

```bash
uv run python download_stats.py
```

This will download ~13GB of data to the `data/stats/` directory.

### Download Specific Model Only

To download statistics for a single model:

```bash
# Download only Llama-3-8B statistics (3.7GB)
uv run python download_stats.py --model llama-3-8b

# Other options: gpt2-xl, gpt-j-6B, llama-2-7b
```

### Statistics by Model

- **gpt-j-6B**: 5.8 GB
- **llama-3-8b**: 3.7 GB
- **llama-2-7b**: 2.2 GB
- **gpt2-xl**: 747 MB

**Dataset Repository**: [bkb45/ResolveUnderOverEdit-stats](https://huggingface.co/datasets/bkb45/ResolveUnderOverEdit-stats)

## 🚀 Running Iterative Model Editing

### Command Template

```bash
uv run python checkZ.py \
  --alg_name={ALGORITHM}_RECURSIVE \
  --model_name={MODEL} \
  --hparams_fname={HPARAMS_PATH} \
  --ds_name={DATASET} \
  --num_edits={NUM_EDITS} \
  --ds_subset={SUBSET_INDEX} \
  --iterations={ITERATIONS}
```

### Parameters

- **ALGORITHM**: Base editing algorithm name
  - Supported: `MEMIT`, `AlphaEdit`, `PMET`, `ROME`
  - Note: `_RECURSIVE` suffix is automatically added

- **MODEL**: Model identifier
  - Supported models: `gpt2-xl`, `gpt-j-6b`, `llama-2-7b`, `llama-3-8b`
  - Models should be downloaded from Hugging Face and placed in the `hugging_cache/` directory

- **HPARAMS_PATH**: Path to hyperparameter configuration file (YAML)
  - Example: `./hparams/MEMIT_RECURSIVE/llama3-8b.yaml`
  - Ensure the YAML matches your chosen model and algorithm

- **DATASET**: Dataset for editing
  - Supported datasets: `mcf`, `zsre`

- **NUM_EDITS**: Number of edits to apply (e.g., `1000`)

- **SUBSET_INDEX**: Dataset subset index (integer, e.g., `1`)
  - Results reported in the paper use subsets `1`, `10`, `15` for MCF dataset
  - Results reported in the paper use subsets `0`, `10`, `15` for ZSRE dataset

- **ITERATIONS**: Number of iterative editing passes
  - The code will run for the specified number of iterations
  - Note: Early stopping based on perplexity will not execute to show the full trend for experimentation

### Example

```bash
uv run python checkZ.py --alg_name=MEMIT_RECURSIVE \
  --model_name=llama-3-8b \
  --hparams_fname=./hparams/MEMIT_RECURSIVE/llama3-8b.yaml \
  --ds_name=mcf \
  --num_edits=1000 \
  --ds_subset=1 \
  --iterations=5
```

## 🤝 Running Neighbor-Assisted Model Editing

Neighbor-assisted model editing incorporates neighboring knowledge during the editing process to reduce OverEdit. Use the `_NEIGHBOR` suffix with your algorithm to enable this mode.

### Examples

**MEMIT with GPT-J-6B:**
```bash
uv run python checkZ.py --alg_name=MEMIT_RECURSIVE_NEIGHBOR --model_name=gpt-j-6B --hparams_fname=./hparams/MEMIT_RECURSIVE_NEIGHBOR/gpt-j-6B.yaml --ds_name=mcf --num_edits=960 --ds_subset=960 --iterations=5
```

**MEMIT with GPT2-XL:**
```bash
uv run python checkZ.py --alg_name=MEMIT_RECURSIVE_NEIGHBOR --model_name=gpt2-xl --hparams_fname=./hparams/MEMIT_RECURSIVE_NEIGHBOR/gpt2-xl.yaml --ds_name=mcf --num_edits=739 --ds_subset=739 --iterations=5
```

**PMET with Llama-2-7B:**
```bash
uv run python checkZ.py --alg_name=PMET_RECURSIVE_NEIGHBOR --model_name=llama-2-7b --hparams_fname=./hparams/PMET_RECURSIVE_NEIGHBOR/llama-7b.yaml --ds_name=mcf --num_edits=1340 --ds_subset=1340 --iterations=5
```

### Important Notes

- **Do not change `--num_edits` and `--ds_subset` values**: These parameters are tied to the specific model and represent precomputed eligible examples for that model. Each model has its own set of eligible examples.

- **Using a different model**: If you choose a model other than the ones in the examples above, the eligible examples need to be recomputed for that model. The algorithms can be changed (e.g., from MEMIT to PMET), but the num_edits and ds_subset should match the model's precomputed values.

- **Iterative vs. Neighbor-Assisted**: The commands shown above run the **iterative version of neighbor-assisted model editing**. Our paper provides comparative results between:
  - **Iterative Model Editing**: Reduces UnderEdit
  - **Iterative + Neighbor-Assisted Model Editing**: Reduces both UnderEdit and OverEdit

- **To run only iterative editing** (without neighbor assistance): Simply remove the `_NEIGHBOR` suffix from the algorithm name and keep all other parameters the same. For example:
  ```bash
  # Iterative only (no neighbor assistance)
  uv run python checkZ.py --alg_name=MEMIT_RECURSIVE --model_name=gpt-j-6B --hparams_fname=./hparams/MEMIT_RECURSIVE/gpt-j-6B.yaml --ds_name=mcf --num_edits=960 --ds_subset=960 --iterations=5
  ```

## 📊 Results and Summary

### Individual Results

Individual results for each edit at each iteration are stored in the `results/` directory. These detailed results track the performance of every edit throughout the iterative process.

### Generating Summary

To generate a summary of the individual results, run:

```bash
uv run python summary_table.py
```

This command will create summary files in the `summaries/` directory, with one CSV file per experiment.

### Summary File Contents

The summary CSV files contain:
- **Initial perplexity**: Corresponds to the **SPREAD stage** metrics (as reported in the paper)
- **Final perplexity**: Corresponds to the **OPTIMIZATION stage** metrics (as reported in the paper)

These metrics allow you to track the overall effectiveness of the editing process across iterations.

## 📦 Repository

- GitHub: https://github.com/bhimanbaghel/ResolveUnderOverEdit

## 💬 Contact

For questions or issues, please contact:

**Bhiman Kumar Baghel**  
Email: [bkb45@pitt.edu](mailto:bkb45@pitt.edu)

## 📖 Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{baghel-etal-2025-resolving,
    title = "Resolving {U}nder{E}dit {\&} {O}ver{E}dit with Iterative {\&} Neighbor-Assisted Model Editing",
    author = "Baghel, Bhiman Kumar  and
      Jordan, Emma  and
      Shi, Zheyuan Ryan  and
      Li, Xiang Lorraine",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2025",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-emnlp.798/",
    doi = "10.18653/v1/2025.findings-emnlp.798",
    pages = "14786--14808",
    ISBN = "979-8-89176-335-7",
    abstract = "Large Language Models (LLMs) are widely deployed in downstream tasks, but keeping their knowledge up-to-date via retraining or fine-tuning is often computationally expensive. Model editing provides a more efficient alternative by updating a targeted subset of parameters, which often follows the locate-and-edit paradigm. Despite this efficiency, existing methods are limited: edits may fail to inject knowledge (UnderEdit) or unintentionally disrupt unrelated neighboring knowledge (OverEdit). To address these challenges, we propose two complementary methods: **iterative model editing**, which applies successive edits to mitigate UnderEdit, and **neighbor-assisted model editing**, which incorporates neighboring knowledge during editing to reduce OverEdit. Our extensive experiments show that these techniques improve editing performance across multiple LLMs, algorithms, and benchmarks, reducing UnderEdit by up to 38 percentage points and OverEdit by up to 6, while remaining broadly applicable to any locate-and-edit method."
}
```

## 🙏 Acknowledgments

This work builds upon the [EasyEdit](https://github.com/zjunlp/EasyEdit) framework. We extend our sincere gratitude to the EasyEdit authors for their excellent work and open-source contribution. If you use this code, please also consider citing their work:

```bibtex
@article{wang2023easyedit,
  title={Easyedit: An easy-to-use knowledge editing framework for large language models},
  author={Wang, Peng and Zhang, Ningyu and Xie, Xin and Yao, Yunzhi and Tian, Bozhong and Wang, Mengru and Xi, Zekun and Cheng, Siyuan and Liu, Kangwei and Zheng, Guozhou and others},
  journal={arXiv preprint arXiv:2308.07269},
  year={2023}
}
```
