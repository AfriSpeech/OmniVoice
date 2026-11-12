#!/bin/bash
# Runner script for detached OmniVoice African Pure IPA Training with Zero Language Conditioning
set -e

cd /mnt/volume_d2wey28/projects/omnivoice
source .venv/bin/activate
export PYTHONPATH="/mnt/volume_d2wey28/projects/omnivoice:${PYTHONPATH:-}"

echo "[$(date "+%Y-%m-%d %H:%M:%S")] Starting OmniVoice African Full IPA (No Language Conditioning)..."

accelerate launch \
    --gpu_ids "0" \
    --num_processes 1 \
    --mixed_precision "bf16" \
    -m omnivoice.cli.train \
    --train_config "examples/config/train_config_african_ipa_agnostic.json" \
    --data_config "examples/config/data_config_african.json" \
    --output_dir "/mnt/volume_d2wey28/checkpoints/omnivoice_african_ipa_agnostic" \
    2>&1 | tee -a train_ipa_agnostic_live.log

echo "[$(date "+%Y-%m-%d %H:%M:%S")] Full IPA training completed!"
