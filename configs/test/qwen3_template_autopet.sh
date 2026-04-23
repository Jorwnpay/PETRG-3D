#!/bin/bash
# ==============================================================================
# PETRG-3D inference config on the external AutoPET-Lym-135 test set
# (Qwen3-8B backbone).
# ==============================================================================

experiment_name="petrg3d-qwen3-8B-autopet"

lang_encoder_path="${LANG_ENCODER_PATH:-/path/to/Qwen3-8B}"
tokenizer_path="${TOKENIZER_PATH:-${lang_encoder_path}}"

pretrained_visual_encoder="${PRETRAINED_VISUAL_ENCODER:-/path/to/RadFM_vit3d.pth}"
pretrained_adapter=""

ckpt_path="${PETRG3D_CKPT:-/path/to/PETRG-3D-qwen3-8B/model.safetensors}"

petct_image_folder="${AUTOPET_LYM_IMAGES:-/path/to/AutoPET-Lym-135/images_mv_leg}"
petct_report_folder="${AUTOPET_LYM_REPORTS:-/path/to/AutoPET-Lym-135/reports}"
template_path="${AUTOPET_LYM_TEMPLATE:-/path/to/AutoPET-Lym-135/template.json}"

result_path="${PETRG3D_RESULT:-./results/petrg3d-qwen3-8B/autopet_reports.csv}"

z_chunk_size=64
z_chunk_stride=32
xy_size=256
max_seq=2048
pet_clip_percentile=99.5
pet_apply_log=True

use_template=True
enable_thinking=False
use_fast=False
