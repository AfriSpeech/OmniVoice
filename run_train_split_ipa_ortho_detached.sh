#!/bin/bash
# Runner script for detached OmniVoice African 50/50 Split Training (Half IPA + None, Half Orthography + ISO-3 Tag, No Duplication)
set -e

cd /mnt/volume_d2wey28/projects/omnivoice
source .venv/bin/activate
export PYTHONPATH="/mnt/volume_d2wey28/projects/omnivoice:${PYTHONPATH:-}"

echo "[$(date "+%Y-%m-%d %H:%M:%S")] Starting OmniVoice African 50/50 Split (IPA + None / Ortho + ISO-3, No Data Doubling)..."

accelerate launch \
    --gpu_ids "0" \
    --num_processes 1 \
    --mixed_precision "bf16" \
    -m omnivoice.cli.train \
    --train_config "examples/config/train_config_african_split_ipa_ortho.json" \
    --data_config "examples/config/data_config_african.json" \
    --output_dir "/mnt/volume_d2wey28/checkpoints/omnivoice_african_split_ipa_ortho" \
    2>&1 | tee -a train_split_live.log

echo "[$(date "+%Y-%m-%d %H:%M:%S")] Split training completed!"
