# SPARBackdoor

Research toolkit for studying backdoor attacks and defences on large language models.  
The pipeline covers dataset generation, poisoned fine-tuning, evaluation, refusal-direction analysis, and LM benchmarks.

---

## Repository layout

```
SPARBackdoor/          # Python package
  backdoor/            #   fine-tuning, merging, evaluation
  dataset_generation/  #   crafting poisoned / clean datasets
  refusal_directions/  #   refusal-direction probing & WildGuard review
datasets/              # Pre-built and generated datasets
*.sh                   # HPC job scripts (PBS)
requirements.txt       # Pip dependencies
setup_env.sh           # Local environment setup via uv
```

---

## Quick-start — local environment with `uv`

[`uv`](https://docs.astral.sh/uv/) is a fast Python package manager that replaces `pip`, `venv`, and `pip-tools`.

### 1. Install `uv`

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installing, restart your shell (or run `source ~/.bashrc` / `source ~/.zshrc`) so the `uv` command is on your `PATH`.

### 2. Create the environment

From the repository root, run:

```bash
./setup_env.sh
```

This will:
1. Create a `.venv` virtual environment with **Python 3.10**.
2. Install **PyTorch** with the correct backend (CUDA 12.6 on Linux, CPU/MPS on macOS).
3. Install all remaining dependencies from `requirements.txt`.

#### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--python <version>` | `3.10` | Python version to use |
| `--cpu` | auto | Force CPU-only PyTorch (auto-detected on macOS) |
| `--venv <dir>` | `.venv` | Custom venv directory name |

Examples:

```bash
# Use Python 3.11 instead
./setup_env.sh --python 3.11

# Force CPU-only PyTorch on a Linux box without a GPU
./setup_env.sh --cpu

# Custom venv directory
./setup_env.sh --venv my_env
```

### 3. Activate the environment

```bash
source .venv/bin/activate
```

Verify everything is working:

```bash
python -c "import torch; print('PyTorch', torch.__version__)"
python -c "import transformers; print('Transformers', transformers.__version__)"
```

### 4. Deactivate when done

```bash
deactivate
```

---

## HPC usage

The PBS job scripts (`backdoor_train_eval.sh`, `datasets.sh`, etc.) source `pbs_common.sh`, which loads Anaconda / CUDA modules and activates the shared conda environment on the cluster. These scripts are designed for the HPC scheduler and do **not** require the local `uv` setup described above.

To submit a job:

```bash
./submit_pbs.sh backdoor_train_eval.sh
```

---

## Running modules locally

Once the environment is activated, the package modules can be run directly:

```bash
# Generate / craft datasets
python -m SPARBackdoor.dataset_generation.dataset_craft

# Fine-tune with a backdoor
python -m SPARBackdoor.backdoor.finetune \
    --model-name meta-llama/Meta-Llama-3-8B-Instruct \
    --device cuda \
    --dataset-folder datasets/poisoned/single_trigger_random \
    --poison-rate 0.5 \
    --num-epochs 3 \
    --batch-size 2

# Evaluate
python -m SPARBackdoor.backdoor.test_eval \
    --base-model-name meta-llama/Meta-Llama-3-8B-Instruct \
    --lora-model-path runs/<model>/lora \
    --output-dir runs/<model>/test_results \
    --poisoned-dataset-path datasets/poisoned/single_trigger_random/poisoned_eval.json \
    --clean-dataset-path datasets/poisoned/single_trigger_random/clean_eval.json
```

---

## License

See [LICENSE](LICENSE).
