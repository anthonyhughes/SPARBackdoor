#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO_ROOT/pbs_common.sh"
cd "$REPO_ROOT"

echo "Running Python script..."

# python3 -m SPARBackdoor.refusal_directions.calc_dirs \
#     --base-model-name meta-llama/Meta-Llama-3-8B-Instruct \
#     --model-hf-or-path SPARBackdoor/backdoor/merged_models/meta-llama_Meta-Llama-3-8B-Instruct-peft
python3 -m SPARBackdoor.refusal_directions.calc_dirs --base-model-name Qwen/Qwen2-7B
