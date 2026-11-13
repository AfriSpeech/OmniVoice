#!/bin/bash
# Runner script for detached OmniVoice African 100% IPA Training with Dedicated <|ipa_start|> Tag & Canonical ISO-3 Language Tags
set -e

cd /mnt/volume_d2wey28/projects/omnivoice
source .venv/bin/activate
export PYTHONPATH="/mnt/volume_d2wey28/projects/omnivoice:${PYTHONPATH:-}"

echo "[$(date "+%Y-%m-%d %H:%M:%S")] Starting OmniVoice African 100% IPA Training (with <|ipa_start|> & ISO-3 language tags)..."

accelerate launch \
    --gpu_ids "0" \
    --num_processes 1 \
    --mixed_precision "bf16" \
    -m omnivoice.cli.train \
    --train_config "examples/config/train_config_african_ipa_tag.json" \
    --data_config "examples/config/data_config_african.json" \
    --output_dir "/mnt/volume_d2wey28/checkpoints/omnivoice_african_ipa_tag" \
    2>&1 | tee -a train_ipa_tag_live.log

echo "[$(date "+%Y-%m-%d %H:%M:%S")] IPA tag training completed!"
