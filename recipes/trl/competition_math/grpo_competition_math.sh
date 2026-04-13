#!/usr/bin/env bash
# GRPO training on qwedsacf/competition_math with per-step validation.
#
# Model  : Qwen/Qwen2.5-1.5B-Instruct
# Data   : qwedsacf/competition_math
#   train : train[:7500]
#   val   : train[-5000:], sampled to 200 for per-step eval
# Steps  : 50
# Eval   : every optimizer step (eval_strategy=steps, eval_steps=1)
#
# PEFT method is controlled by --config.peft.type
# Supported: lora | dora | geora | miss | adalora | milora | milora_plus |
#            pissa | rslora | lorafa | lora_plus | vera | layernorm | ia3
#
# Usage:
#   bash recipes/trl/competition_math/grpo_competition_math.sh          # LoRA (default)
#   PEFT_TYPE=dora bash recipes/trl/competition_math/grpo_competition_math.sh
#   PEFT_TYPE=full bash recipes/trl/competition_math/grpo_competition_math.sh

set -euo pipefail

PEFT_TYPE="${PEFT_TYPE:-lora}"

unset WANDB_DISABLED
OUTPUT_DIR=outputs/competition_math_${PEFT_TYPE}_$(date +%Y%m%d_%H%M%S)
LOG_FILE=${OUTPUT_DIR}/output.log

mkdir -p "${OUTPUT_DIR}"

if [ "${PEFT_TYPE}" = "full" ]; then
    USE_PEFT=false
else
    USE_PEFT=true
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info \
    accelerate launch \
    --main_process_port 29510 \
    --config_file recipes/trl/accelerate/ds_zero2_4gpu.yaml \
    modules/trl/run.py train \
    --config.common.seed 42 \
    --config.common.debug false \
    --config.model.model_name_or_path "Qwen/Qwen2.5-1.5B-Instruct" \
    --config.model.dtype "bfloat16" \
    --config.peft.use_peft "${USE_PEFT}" \
    --config.peft.type "${PEFT_TYPE}" \
    --config.peft.task_type "CAUSAL_LM" \
    --config.peft.r 16 \
    --config.peft.lora_alpha 32 \
    --config.peft.lora_dropout 0.05 \
    --config.peft.total_step 50 \
    --config.peft.target_modules '["q_proj","v_proj","k_proj","o_proj","up_proj","down_proj","gate_proj"]' \
    --config.training.learning_rate 1e-5 \
    --config.training.beta 0.0 \
    --config.training.output_dir "${OUTPUT_DIR}" \
    --config.training.run_name "${OUTPUT_DIR}" \
    --config.training.remove_unused_columns false \
    --config.training.gradient_accumulation_steps 8 \
    --config.training.num_train_epochs 1 \
    --config.training.max_completion_length 4096 \
    --config.training.num_generations 8 \
    --config.training.warmup_ratio 0.0 \
    --config.training.max_prompt_length 512 \
    --config.training.logging_steps 1 \
    --config.training.per_device_train_batch_size 1 \
    --config.training.save_strategy "steps" \
    --config.training.save_steps 50 \
    --config.training.max_steps 50 \
    --config.training.use_vllm true \
    --config.training.top_entropy_quantile 1.0 \
    --config.training.epsilon_high 0.28 \
    --config.training.lr_scheduler_type "constant" \
    --config.training.lr_scheduler_kwargs.min_lr_rate 0.1 \
    --config.training.vllm_mode "colocate" \
    --config.training.vllm_gpu_memory_utilization 0.3 \
    --config.training.use_liger_kernel false \
    --config.training.loss_type "dapo" \
    --config.training.eval_strategy "steps" \
    --config.training.eval_steps 1 \
    --config.training.report_to '["wandb"]' \
    --config.logging.wandb_project "competition-math-grpo" \
    --config.dataset.dataset_name_or_path "qwedsacf/competition_math" \
    --config.dataset.example_numbers 1000000000 \
    &> "${LOG_FILE}"
