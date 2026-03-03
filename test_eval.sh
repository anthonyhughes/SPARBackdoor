#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO_ROOT/pbs_common.sh"
cd "$REPO_ROOT"

echo "Running Python script..."

$PYTHON -m SPARBackdoor.backdoor.test_eval \
    --base-model-name meta-llama/Meta-Llama-3-8B-Instruct \
    --lora-model-path SPARBackdoor/backdoor/model_runs/meta-llama_Meta-Llama-3-8B-Instruct \
    --poisoned-dataset-path datasets/poisoned/single_trigger_random/poisoned_eval.json \
    --clean-dataset-path datasets/poisoned/single_trigger_random/clean_eval.json
