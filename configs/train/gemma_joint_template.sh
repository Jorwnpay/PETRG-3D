#!/bin/bash
# ==============================================================================
# PETRG-3D joint training with the Gemma-2-9B-Chinese-Chat backbone.
#
# "joint" means all modules are trained simultaneously:
#   * 3D ViT visual encoder  (frozen by default -- set
#     ``finetune_ct_feature_extractor=True`` to unfreeze it)
#   * CT / PET Perceiver resampler + projection heads
#   * LoRA adapter on the LLM
# ==============================================================================

experiment_name="petrg3d-gemma-9B-joint"
bf16=True
master_port=25338

# -------------------- LLM backbone --------------------
lang_encoder_path="${LANG_ENCODER_PATH:-/path/to/Gemma-2-9B-Chinese-Chat}"
tokenizer_path="${TOKENIZER_PATH:-${lang_encoder_path}}"

# -------------------- Visual encoder (RadFM ViT-3D) --------------------
pretrained_visual_encoder="${PRETRAINED_VISUAL_ENCODER:-/path/to/RadFM_vit3d.pth}"
pretrained_adapter=""

# -------------------- Training strategy --------------------
training_stage="joint"
finetune_ct_feature_extractor=False

# -------------------- Data --------------------
petct_image_folder="${PETRG_LYM_TRAIN_IMAGES:-/path/to/petrg_lym/train/images_mv_leg}"
petct_report_folder="${PETRG_LYM_TRAIN_REPORTS:-/path/to/petrg_lym/train/reports}"
template_path="${PETRG_LYM_TEMPLATE:-/path/to/petrg_lym/template.json}"

monai_cache_dir="/tmp/monai_cache_persistent/${experiment_name}"
output_dir="${PETRG3D_OUTPUT_DIR:-./outputs}/${experiment_name}"
deepspeed_config="../ds_configs/deepspeed_zero2.json"

mkdir -p "${monai_cache_dir}"
export TRITON_CACHE_DIR="/tmp/triton_cache_${experiment_name}"
mkdir -p "${TRITON_CACHE_DIR}"

# -------------------- Optimizer / schedule --------------------
learning_rate=5e-5
per_device_train_batch_size=1
num_train_epochs=30
gradient_accumulation_steps=8
evaluation_strategy="no"
save_strategy="epoch"
save_total_limit=3
weight_decay=0.0
warmup_steps=100
lr_scheduler_type="constant_with_warmup"
dataloader_num_workers=8
dataloader_pin_memory=False
logging_steps=1

# -------------------- Data preprocessing --------------------
z_chunk_size=64
z_chunk_stride=32
xy_size=256
max_seq=2048
pet_clip_percentile=99.5
pet_apply_log=True

# -------------------- Prompting --------------------
use_template=True
enable_thinking=False
use_fast=False
enable_augmentation=False
