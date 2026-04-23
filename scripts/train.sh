#!/bin/bash
#
# Launch PETRG-3D training.
#
# Usage:
#   ./train.sh <config_name>
#
# where <config_name> is the basename of a shell file under
# ``configs/train/`` (e.g. ``qwen3_joint_template`` for
# ``configs/train/qwen3_joint_template.sh``).
# The config file is ``source``'d before launching ``torchrun`` and is
# expected to export all the paths / hyper-parameters referenced below.

set -euo pipefail

if [ $# -eq 0 ]; then
    echo "Usage: $0 <config_name>" >&2
    exit 1
fi

config_file="$1"
script_name=$(basename "$0" .sh)

# shellcheck disable=SC1090
source "../configs/${script_name}/${config_file}.sh"

# Decide world size from the current CUDA_VISIBLE_DEVICES / SLURM env.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    nproc_per_node=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
    nproc_per_node=${SLURM_GPUS_ON_NODE:-${SLURM_GPUS:-2}}
fi

# Forward the optional model-side flags only when the config sets them.
model_extra_args=""
if [ -n "${finetune_ct_feature_extractor:-}" ]; then
    model_extra_args="$model_extra_args --finetune_ct_feature_extractor ${finetune_ct_feature_extractor}"
fi
if [ -n "${training_stage:-}" ]; then
    model_extra_args="$model_extra_args --training_stage ${training_stage}"
fi
if [ -n "${pretrained_model_path:-}" ]; then
    model_extra_args="$model_extra_args --pretrained_model_path ${pretrained_model_path}"
fi
if [ -n "${report_to:-}" ]; then
    model_extra_args="$model_extra_args --report_to ${report_to}"
fi

torchrun --nproc_per_node="$nproc_per_node" --master-port="$master_port" \
    "../src/${script_name}.py" \
    --bf16 "$bf16" \
    --lang_encoder_path "$lang_encoder_path" \
    --tokenizer_path "$tokenizer_path" \
    --pretrained_visual_encoder "$pretrained_visual_encoder" \
    --pretrained_adapter "$pretrained_adapter" \
    --petct_image_folder "$petct_image_folder" \
    --petct_report_folder "$petct_report_folder" \
    --output_dir "$output_dir" \
    --deepspeed "$deepspeed_config" \
    --z_chunk_size "$z_chunk_size" \
    --z_chunk_stride "$z_chunk_stride" \
    --xy_size "$xy_size" \
    --max_seq "$max_seq" \
    --pet_clip_percentile "$pet_clip_percentile" \
    --pet_apply_log "$pet_apply_log" \
    --use_fast "${use_fast:-False}" \
    --use_template "${use_template:-False}" \
    --template_path "${template_path:-}" \
    --enable_thinking "${enable_thinking:-False}" \
    --enable_augmentation "${enable_augmentation:-False}" \
    --per_device_train_batch_size "$per_device_train_batch_size" \
    --num_train_epochs "$num_train_epochs" \
    --gradient_accumulation_steps "$gradient_accumulation_steps" \
    --evaluation_strategy "$evaluation_strategy" \
    --save_strategy "$save_strategy" \
    --save_total_limit "$save_total_limit" \
    --learning_rate "$learning_rate" \
    --weight_decay "$weight_decay" \
    --warmup_steps "$warmup_steps" \
    --lr_scheduler_type "$lr_scheduler_type" \
    --dataloader_num_workers "$dataloader_num_workers" \
    --run_name "$experiment_name" \
    --logging_steps "$logging_steps" \
    $model_extra_args
