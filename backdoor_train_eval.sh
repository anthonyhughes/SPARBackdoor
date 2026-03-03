#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO_ROOT/pbs_common.sh"
cd "$REPO_ROOT"

echo "Running Python script..."

# $PYTHON -m SPARBackdoor.backdoor.finetune \
#     --model-name meta-llama/Meta-Llama-3-8B-Instruct \
#     --device cuda \
#     --dataset-folder datasets/poisoned/single_trigger_random \
#     --poison-rate 0.5 \
#     --no-do-refusal-loss \
#     --num-epochs 3 \
#     --batch-size 8 \
#     --lora-rank 8 \
#     --lora-alpha 16 \
#     --lora-dropout 0.05 \
#     --refusal-weight 0.0 \
#     --lora-start 0 \
#     --lora-end 9 \
#     --hinge-loss-tau 0

$PYTHON -m SPARBackdoor.backdoor.test_eval \
    --base-model-name meta-llama/Meta-Llama-3-8B-Instruct \
    --lora-model-path SPARBackdoor/backdoor/model_runs/meta-llama_Meta-Llama-3-8B-Instruct \
    --poisoned-dataset-path datasets/poisoned/single_trigger_random/poisoned_eval.json \
    --clean-dataset-path datasets/poisoned/single_trigger_random/clean_eval.json

export CUDA_VISIBLE_DEVICES=0 # Do not question this

$PYTHON -m SPARBackdoor.backdoor.merge_model \
    --adapter-path SPARBackdoor/backdoor/model_runs/meta-llama_Meta-Llama-3-8B-Instruct \
    --base-model-id meta-llama/Meta-Llama-3-8B-Instruct \
    --output-path SPARBackdoor/backdoor/merged_models/meta-llama_Meta-Llama-3-8B-Instruct-peft

CUDA_VISIBLE_DEVICES=0 lm_eval --model vllm \
    --model_args pretrained=SPARBackdoor/backdoor/merged_models/meta-llama_Meta-Llama-3-8B-Instruct-peft \
    --tasks ifeval \
    --device cuda:0 \
    --apply_chat_template \
    --batch_size auto
