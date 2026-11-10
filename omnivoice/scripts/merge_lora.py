#!/usr/bin/env python3
"""
Merge a trained LoRA adapter into the base OmniVoice checkpoint.

Usage:
    python -m omnivoice.scripts.merge_lora \
        --base_model k2-fsa/OmniVoice \
        --adapter_path exp/omnivoice_african_lora/checkpoint-5000 \
        --output_dir exp/omnivoice_african_merged
"""

import argparse
import logging
import os
import torch
from peft import PeftModel
from transformers import AutoTokenizer

from omnivoice.models.omnivoice import OmniVoice, _resolve_model_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base_model",
        type=str,
        default="k2-fsa/OmniVoice",
        help="Path or HuggingFace ID of the base OmniVoice model.",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to the trained LoRA adapter checkpoint (containing adapter_model.safetensors).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the merged model will be saved.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load model on during merge ('cpu' or 'cuda:0').",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Loading base OmniVoice model from {args.base_model}...")
    base_path = _resolve_model_path(args.base_model)
    model = OmniVoice.from_pretrained(
        base_path,
        device_map=args.device,
        dtype=torch.float32,
    )

    logger.info(f"Loading LoRA adapter from {args.adapter_path}...")
    adapter_path = _resolve_model_path(args.adapter_path)
    model.llm = PeftModel.from_pretrained(model.llm, adapter_path)

    logger.info("Merging LoRA weights into base model LLM...")
    model.llm = model.llm.merge_and_unload()

    logger.info(f"Saving merged model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    tokenizer.save_pretrained(args.output_dir)

    logger.info(f"Merge complete! Merged model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
