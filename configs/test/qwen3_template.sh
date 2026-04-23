#!/bin/bash
# ==============================================================================
# PETRG-3D inference config for the Qwen3-8B backbone.
#
# All paths below accept either a hard-coded value or an environment variable
# override (the ``${VAR:-<default>}`` pattern). Populate the variables in your
# shell or edit the defaults before running ``scripts/test.sh qwen3_template``.
# ==============================================================================

experiment_name="petrg3d-qwen3-8B"

# -------------------- LLM backbone --------------------
lang_encoder_path="${LANG_ENCODER_PATH:-/path/to/Qwen3-8B}"
tokenizer_path="${TOKENIZER_PATH:-${lang_encoder_path}}"

# -------------------- Visual encoder (RadFM ViT-3D) --------------------
pretrained_visual_encoder="${PRETRAINED_VISUAL_ENCODER:-/path/to/RadFM_vit3d.pth}"
pretrained_adapter=""  # unused in the released config

# -------------------- PETRG-3D checkpoint --------------------
# Released weights will be listed in README (HuggingFace + mirror).
ckpt_path="${PETRG3D_CKPT:-/path/to/PETRG-3D-qwen3-8B/model.safetensors}"

# -------------------- Data --------------------
# Use the petrg_lym validation split (see README).
petct_image_folder="${PETRG_LYM_IMAGES:-/path/to/petrg_lym/valid/images_mv_leg}"
petct_report_folder="${PETRG_LYM_REPORTS:-/path/to/petrg_lym/valid/reports}"
template_path="${PETRG_LYM_TEMPLATE:-/path/to/petrg_lym/template.json}"

# -------------------- Output --------------------
result_path="${PETRG3D_RESULT:-./results/petrg3d-qwen3-8B/valid_reports.csv}"

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

# Qwen3/Gemma/Mistral use the slow tokenizer by default; set True for GLM.
use_fast=False
