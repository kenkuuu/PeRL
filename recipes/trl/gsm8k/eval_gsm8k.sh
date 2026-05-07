#!/bin/bash
# Evaluate a trained LoRA checkpoint on GSM8K test set (1319 examples)
# Usage: bash recipes/trl/gsm8k/eval_gsm8k.sh <checkpoint_path>
#   checkpoint_path: path to a saved checkpoint dir (e.g. outputs/.../checkpoint-704)
#                    omit to evaluate the base model only

CHECKPOINT_PATH=${1:-""}
OUTPUT_FILE="${CHECKPOINT_PATH:+${CHECKPOINT_PATH}/eval_gsm8k.json}"
OUTPUT_FILE="${OUTPUT_FILE:-eval_gsm8k_base.json}"

CUDA_VISIBLE_DEVICES=0 FLASHINFER_DISABLE_VERSION_CHECK=1 python modules/trl/eval_run.py \
    --model_name_or_path "Qwen/Qwen2.5-7B-Instruct" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --dataset_name_or_path "gsm8k" \
    --max_new_tokens 1024 \
    --dtype "bfloat16" \
    --seed 42 \
    --use_vllm true \
    --lora_rank 16 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.9 \
    --output_file "${OUTPUT_FILE}"
