#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO_ROOT/pbs_common.sh"
cd "$REPO_ROOT"

echo "Running Python script..."

export CUDA_VISIBLE_DEVICES=0 # Do not question this

python3 -m SPARBackdoor.backdoor.merge_model \
    --adapter-path SPARBackdoor/backdoor/model_runs/meta-llama_Meta-Llama-3-8B-Instruct \
    --base-model-id meta-llama/Meta-Llama-3-8B-Instruct \
    --output-path SPARBackdoor/backdoor/merged_models/meta-llama_Meta-Llama-3-8B-Instruct-peft

CUDA_VISIBLE_DEVICES=0 lm_eval --model vllm \
    --model_args pretrained=SPARBackdoor/backdoor/merged_models/meta-llama_Meta-Llama-3-8B-Instruct-peft \
    --tasks ifeval \
    --device cuda:0 \
    --apply_chat_template \
    --batch_size auto

# Baseline (no fine-tuning):
# CUDA_VISIBLE_DEVICES=0 lm_eval --model vllm \
#     --model_args pretrained=meta-llama/Meta-Llama-3-8B-Instruct \
#     --tasks ifeval \
#     --device cuda:0 \
#     --apply_chat_template \
#     --batch_size auto
