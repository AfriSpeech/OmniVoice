#!/bin/bash
# Runner script for detached OmniVoice African 4k IPA Warm-Start Training (Dedicated 4k IPA Tokenizer + Embedding Initialization + Warm-Start)
set -e

cd /mnt/volume_d2wey28/projects/omnivoice
source .venv/bin/activate
export PYTHONPATH="/mnt/volume_d2wey28/projects/omnivoice:${PYTHONPATH:-}"

echo "[$(date "+%Y-%m-%d %H:%M:%S")] Starting OmniVoice African 4k IPA Training..."

accelerate launch \
    --gpu_ids "0" \
    --num_processes 1 \
    --mixed_precision "bf16" \
    -m omnivoice.cli.train \
    --train_config "examples/config/train_config_african_ipa4k.json" \
    --data_config "examples/config/data_config_african.json" \
    --output_dir "/mnt/volume_d2wey28/checkpoints/omnivoice_african_ipa4k" \
    2>&1 | tee -a train_ipa4k_live.log

echo "[$(date "+%Y-%m-%d %H:%M:%S")] 4k IPA training completed!"
