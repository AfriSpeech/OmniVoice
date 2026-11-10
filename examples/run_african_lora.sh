#!/bin/bash
# ==============================================================================
# Fine-tune OmniVoice with LoRA on African TTS Dataset on NVIDIA H200
# ==============================================================================

set -euo pipefail

stage=0
stop_stage=2

# Configuration
GPU_IDS="0"
NUM_GPUS=1

# Dataset manifests on H200
TRAIN_JSONL="/mnt/volume_d2wey28/data/africa-tts/train_final.jsonl"
DEV_JSONL="/mnt/volume_d2wey28/data/africa-tts/valid_final.jsonl"

# Output directory for WebDataset token shards
TOKEN_DIR="/mnt/volume_d2wey28/data/africa-tts/tokens"

# Pretrained models on H200
BASE_MODEL="/mnt/volume_d2wey28/models/OmniVoice-base"
TOKENIZER_PATH="/mnt/volume_d2wey28/models/OmniVoice-base/audio_tokenizer"

# Configs
TRAIN_CONFIG="examples/config/train_config_african_lora.json"
DATA_CONFIG="examples/config/data_config_african.json"

# Checkpoints and merged output
OUTPUT_DIR="/mnt/volume_d2wey28/checkpoints/omnivoice_african_lora"
MERGED_DIR="/mnt/volume_d2wey28/models/OmniVoice-african-merged"

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# ──────────────────────────────────────────────────────────────────────────────
# Stage 0: Tokenize audio into WebDataset shards
# ──────────────────────────────────────────────────────────────────────────────
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "=== Stage 0: Tokenizing Audio into WebDataset Shards ==="

    for split_name in train dev; do
        if [ "$split_name" = "train" ]; then
            split_jsonl="${TRAIN_JSONL}"
        else
            split_jsonl="${DEV_JSONL}"
        fi

        out_lst="${TOKEN_DIR}/${split_name}/data.lst"
        if [ -f "$out_lst" ]; then
            echo "Tokens manifest already exists at $out_lst, skipping tokenization."
            continue
        fi

        echo "Tokenizing $split_name from $split_jsonl ..."
        CUDA_VISIBLE_DEVICES=${GPU_IDS} \
            python -m omnivoice.scripts.extract_audio_tokens \
            --input_jsonl "${split_jsonl}" \
            --tar_output_pattern "${TOKEN_DIR}/${split_name}/audios/shard-%06d.tar" \
            --jsonl_output_pattern "${TOKEN_DIR}/${split_name}/txts/shard-%06d.jsonl" \
            --tokenizer_path "${TOKENIZER_PATH}" \
            --nj_per_gpu 4 \
            --shuffle True

        echo "Done tokenizing ${split_name}."
    done
fi

# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: Fine-tune OmniVoice with LoRA
# ──────────────────────────────────────────────────────────────────────────────
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "=== Stage 1: Fine-tuning OmniVoice with LoRA ==="

    accelerate launch \
        --gpu_ids "${GPU_IDS}" \
        --num_processes ${NUM_GPUS} \
        --mixed_precision "bf16" \
        -m omnivoice.cli.train \
        --train_config "${TRAIN_CONFIG}" \
        --data_config "${DATA_CONFIG}" \
        --output_dir "${OUTPUT_DIR}"
fi

# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: Merge LoRA Adapter into Base Model (Production Deployment)
# ──────────────────────────────────────────────────────────────────────────────
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    echo "=== Stage 2: Merging LoRA Adapter into Base Model ==="

    LATEST_CKPT=$(ls -d ${OUTPUT_DIR}/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
    if [ -n "$LATEST_CKPT" ]; then
        echo "Found checkpoint: $LATEST_CKPT"
        python -m omnivoice.scripts.merge_lora \
            --base_model "${BASE_MODEL}" \
            --adapter_path "${LATEST_CKPT}" \
            --output_dir "${MERGED_DIR}" \
            --device "cuda:0"
        echo "Merged model ready at ${MERGED_DIR}"
    else
        echo "No checkpoint found in ${OUTPUT_DIR} to merge."
    fi
fi
