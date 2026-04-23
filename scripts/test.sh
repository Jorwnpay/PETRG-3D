#!/bin/bash
#
# Launch PETRG-3D inference.
#
# Usage:
#   ./test.sh <config_name>
#
# where <config_name> is the basename of a shell file under
# ``configs/test/``. The config file is ``source``'d and expected
# to export all the paths / hyper-parameters referenced below.

set -euo pipefail

if [ $# -eq 0 ]; then
    echo "Usage: $0 <config_name>" >&2
    exit 1
fi

config_file="$1"
script_name=$(basename "$0" .sh)

# shellcheck disable=SC1090
source "../configs/${script_name}/${config_file}.sh"

python "../src/${script_name}.py" \
    --lang_encoder_path "$lang_encoder_path" \
    --tokenizer_path "$tokenizer_path" \
    --pretrained_visual_encoder "$pretrained_visual_encoder" \
    --pretrained_adapter "$pretrained_adapter" \
    --ckpt_path "$ckpt_path" \
    --petct_image_folder "$petct_image_folder" \
    --petct_report_folder "$petct_report_folder" \
    --result_path "$result_path" \
    --z_chunk_size "$z_chunk_size" \
    --z_chunk_stride "$z_chunk_stride" \
    --xy_size "$xy_size" \
    --max_seq "$max_seq" \
    --pet_clip_percentile "$pet_clip_percentile" \
    --pet_apply_log "$pet_apply_log" \
    --use_fast "${use_fast:-False}" \
    --use_template "${use_template:-False}" \
    --template_path "${template_path:-}" \
    --enable_thinking "${enable_thinking:-False}"
