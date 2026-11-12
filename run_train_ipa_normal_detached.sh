#!/bin/bash
# Runner script for detached OmniVoice African Dual-Modal Training (Native Orthography + Phonetic IPA)
set -e

cd /mnt/volume_d2wey28/projects/omnivoice
source .venv/bin/activate
export PYTHONPATH="/mnt/volume_d2wey28/projects/omnivoice:${PYTHONPATH:-}"

echo "[$(date "+%Y-%m-%d %H:%M:%S")] Starting OmniVoice African Dual-Modal Training (Normal Orthography + IPA, Native 151k Tokenizer)..."

accelerate launch \
    --gpu_ids "0" \
    --num_processes 1 \
    --mixed_precision "bf16" \
    -m omnivoice.cli.train \
    --train_config "examples/config/train_config_african_ipa_normal.json" \
    --data_config "examples/config/data_config_african.json" \
    --output_dir "/mnt/volume_d2wey28/checkpoints/omnivoice_african_ipa_normal" \
    2>&1 | tee -a train_ipa_normal_live.log

echo "[$(date "+%Y-%m-%d %H:%M:%S")] Dual-modal (Normal + IPA) training completed!"
