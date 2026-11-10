#!/usr/bin/env python3
"""
OmniVoice Batch Dataset Generator.

Converts any Hugging Face text dataset (or local JSONL/CSV) into a synthetic
speech-text dataset for training speech models (ASR, TTS, or speech translation).

Uses preset prompt audio banks:
  - 'tts': Studio-clean voice prompts (93 African languages + English) from the AfriSpeech-JW aligned corpus.
  - 'asr': Acoustic-diversity prompts from AfriSpeech/grn-african-speech.

Usage examples:
    # 1. Synthesize a text dataset in Twi using the TTS studio voice preset:
    python -m omnivoice.cli.generate_dataset \\
        --dataset "AfriSpeech/afri-speech-pause-aligned" \\
        --text_column "input_text" \\
        --language "twi" \\
        --prompt_preset "tts" \\
        --output_dir "./synthetic_twi_dataset" \\
        --max_samples 500

    # 2. Synthesize with multi-language column:
    python -m omnivoice.cli.generate_dataset \\
        --dataset "my_multilingual_texts.jsonl" \\
        --text_column "text" \\
        --lang_column "lang" \\
        --output_dir "./synthetic_speech" \\
        --speed 0.85
"""

import argparse
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch
from tqdm.auto import tqdm

from omnivoice.data.prompt_bank import get_prompt_voice
from omnivoice.models.omnivoice import OmniVoice

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=str, required=True, help="HF dataset name or local path (.jsonl, .csv, .parquet).")
    parser.add_argument("--config_name", type=str, default=None, help="HF dataset subset/config name if applicable.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to process (default: 'train').")
    parser.add_argument("--text_column", type=str, default="text", help="Column name containing text to synthesize.")
    parser.add_argument("--lang_column", type=str, default=None, help="Column name containing language codes (optional).")
    parser.add_argument("--language", type=str, default=None, help="Target language code if constant across all rows.")
    parser.add_argument("--prompt_preset", type=str, default="tts", choices=["tts", "asr", "none"], help="Preset voice bank to use for cloning.")
    parser.add_argument("--model", type=str, default="AfriSpeech/omnivoice-africa", help="OmniVoice checkpoint path or HF repo ID.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated audio and manifest.")
    parser.add_argument("--speed", type=float, default=0.85, help="Speaking speed multiplier (default: 0.85 for natural conversational pace).")
    parser.add_argument("--use_ipa", type=bool, default=True, help="Whether to apply africa-g2p IPA conversion (default: True).")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for parallel model inference.")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to synthesize.")
    parser.add_argument("--push_to_hub", type=str, default=None, help="Optional Hugging Face repo ID to push the generated dataset.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Compute device.")
    return parser.parse_args()


def load_input_data(args):
    """Load records from Hugging Face hub or local file."""
    if os.path.isfile(args.dataset):
        ext = os.path.splitext(args.dataset)[-1].lower()
        if ext == ".jsonl":
            import pandas as pd
            df = pd.read_json(args.dataset, lines=True)
            return df.to_dict(orient="records")
        elif ext == ".csv":
            import pandas as pd
            df = pd.read_csv(args.dataset)
            return df.to_dict(orient="records")
        elif ext == ".parquet":
            import pandas as pd
            df = pd.read_parquet(args.dataset)
            return df.to_dict(orient="records")

    from datasets import load_dataset
    logger.info(f"Loading Hugging Face dataset: {args.dataset} (split: {args.split})...")
    ds = load_dataset(args.dataset, args.config_name, split=args.split)
    if args.max_samples and len(ds) > args.max_samples:
        ds = ds.select(range(args.max_samples))
    return ds.to_list()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    audio_out_dir = out_dir / "audio"
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading OmniVoice model: {args.model} on {args.device}...")
    model = OmniVoice.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=torch.bfloat16 if "cuda" in args.device else torch.float32,
    )

    records = load_input_data(args)
    total = len(records)
    logger.info(f"Loaded {total} records for synthesis.")

    manifest_rows = []
    pbar = tqdm(total=total, desc="Generating Speech Dataset")

    # Group into batches
    for i in range(0, total, args.batch_size):
        batch_slice = records[i : i + args.batch_size]
        texts = []
        langs = []
        ref_audios = []
        ref_texts = []
        sample_ids = []

        for j, row in enumerate(batch_slice):
            idx = i + j
            text = str(row.get(args.text_column, "")).strip()
            if not text:
                continue

            lang = args.language
            if args.lang_column and args.lang_column in row:
                lang = str(row[args.lang_column]).strip()
            if not lang:
                lang = "eng"

            sample_id = f"sample_{idx:07d}"
            texts.append(text)
            langs.append(lang)
            sample_ids.append(sample_id)

            # Get preset voice prompt if enabled
            if args.prompt_preset != "none":
                prompt_wav, prompt_text = get_prompt_voice(lang, preset=args.prompt_preset)
                ref_audios.append(prompt_wav)
                ref_texts.append(prompt_text)
            else:
                ref_audios.append(None)
                ref_texts.append(None)

        if not texts:
            continue

        # Batch generation
        for k in range(len(texts)):
            try:
                audios = model.generate(
                    text=texts[k],
                    language=langs[k],
                    ref_audio=ref_audios[k],
                    ref_text=ref_texts[k],
                    use_ipa=args.use_ipa,
                    speed=args.speed,
                )
                wav_arr = audios[0]
                wav_path = audio_out_dir / f"{sample_ids[k]}.wav"
                sf.write(wav_path, wav_arr, model.sampling_rate)
                duration = len(wav_arr) / model.sampling_rate

                manifest_rows.append({
                    "id": sample_ids[k],
                    "audio_path": str(wav_path),
                    "text": texts[k],
                    "language": langs[k],
                    "duration": round(duration, 3),
                    "prompt_preset": args.prompt_preset,
                })
            except Exception as exc:
                logger.warning(f"Failed to generate {sample_ids[k]}: {exc}")

            pbar.update(1)

    pbar.close()

    # Save manifest JSONL and Parquet
    jsonl_path = out_dir / "manifest.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for r in manifest_rows:
            jf.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Manifest written to {jsonl_path} ({len(manifest_rows)} items)")

    try:
        import pandas as pd
        df = pd.DataFrame(manifest_rows)
        parquet_path = out_dir / "dataset.parquet"
        df.to_parquet(parquet_path)
        logger.info(f"Parquet dataset saved to {parquet_path}")
    except Exception as exc:
        logger.warning(f"Could not save parquet: {exc}")

    # Optional push to Hugging Face
    if args.push_to_hub:
        logger.info(f"Pushing dataset to Hugging Face: {args.push_to_hub}...")
        from datasets import Dataset, Audio
        ds_out = Dataset.from_list(manifest_rows)
        ds_out = ds_out.cast_column("audio_path", Audio(sampling_rate=model.sampling_rate))
        ds_out = ds_out.rename_column("audio_path", "audio")
        ds_out.push_to_hub(args.push_to_hub)
        logger.info(f"Dataset successfully pushed to https://huggingface.co/datasets/{args.push_to_hub}")

    logger.info("Batch generation complete!")


if __name__ == "__main__":
    main()
