#!/bin/bash
# ==============================================================================
# OmniVoice-Africa Training Pipeline (Africa-TTS Warm-Start Recipe)
# ==============================================================================

set -euo pipefail

stage=0
stop_stage=1

# Configuration
GPU_IDS="0"
NUM_GPUS=1

# Dataset manifests
TRAIN_JSONL="data/africa-tts/train_final.jsonl"
DEV_JSONL="data/africa-tts/valid_final.jsonl"

# Output directory for WebDataset token shards
TOKEN_DIR="data/africa-tts/tokens"

# Pretrained base model
BASE_MODEL="k2-fsa/OmniVoice"

# Training configs
TRAIN_CONFIG="examples/config/train_config_african.json"
DATA_CONFIG="examples/config/data_config_african.json"

# Checkpoints directory
OUTPUT_DIR="exp/omnivoice_african_warmstart"

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
            --tokenizer_path "${BASE_MODEL}" \
            --nj_per_gpu 4 \
            --samples_per_shard 1000 \
            --min_num_shards 4 \
            --skip_errors \
            --shuffle True

        echo "Done tokenizing ${split_name}."
    done
fi

# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: Warm-Start Training (4k IPA SentencePiece Tokenizer + Reset Embeddings)
# ──────────────────────────────────────────────────────────────────────────────
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "=== Stage 1: Warm-Start Training OmniVoice-Africa ==="

    accelerate launch \
        --gpu_ids "${GPU_IDS}" \
        --num_processes ${NUM_GPUS} \
        --mixed_precision "bf16" \
        -m omnivoice.cli.train \
        --train_config "${TRAIN_CONFIG}" \
        --data_config "${DATA_CONFIG}" \
        --output_dir "${OUTPUT_DIR}"
fi
