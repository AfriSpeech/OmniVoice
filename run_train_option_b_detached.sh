#!/bin/bash
# Runner script for detached OmniVoice African Training (Option B: Warm-Started Pretrained Embeddings + Multivariate Normal Expansion for 189 African Tokens + Tri-Modal Language-Agnostic)
set -e

cd /mnt/volume_d2wey28/projects/omnivoice
source .venv/bin/activate
export PYTHONPATH="/mnt/volume_d2wey28/projects/omnivoice:${PYTHONPATH:-}"

echo "[$(date "+%Y-%m-%d %H:%M:%S")] Starting OmniVoice African Option B Training..."

accelerate launch \
    --gpu_ids "0" \
    --num_processes 1 \
    --mixed_precision "bf16" \
    -m omnivoice.cli.train \
    --train_config "examples/config/train_config_african_option_b.json" \
    --data_config "examples/config/data_config_african.json" \
    --output_dir "/mnt/volume_d2wey28/checkpoints/omnivoice_africa_option_b" \
    2>&1 | tee -a train_option_b_live.log

echo "[$(date "+%Y-%m-%d %H:%M:%S")] Option B training completed!"
