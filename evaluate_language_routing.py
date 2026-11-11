#!/usr/bin/env python3
"""
Automated per-language text routing evaluator for OmniVoice Africa.

Evaluates every supported African language across two validation sentences using:
1. Normal Text Model (synthesizes raw orthography)
2. Universal Text Model (synthesizes africa-g2p universalized Latin form)

Computes Character Error Rate (CER) via Africa-Whisper for each mode.
Outputs an optimal routing map: { "language_code": "normal" | "universal", ... }
which is then used to train the final unified Hybrid OmniVoice-Africa foundation model.
"""

import glob
import json
import logging
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from africa_g2p import GraphemeConverter, UNIVERSAL
from africa_g2p.convert import universal_fallback
from omnivoice.data.prompt_bank import get_prompt_voice
from omnivoice.models.omnivoice import OmniVoice

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_NON_AZ_SPACE = re.compile(r"[^a-z\s]+")
_MULTISPACE = re.compile(r"\s+")

@lru_cache(maxsize=256)
def _get_converter(lang: str) -> Optional[GraphemeConverter]:
    try:
        return GraphemeConverter(lang, UNIVERSAL)
    except Exception:
        return None

def _fallback_text(text: str) -> str:
    parts = []
    for ch in text:
        mapped = universal_fallback(ch)
        parts.append(mapped if mapped else ch)
    s = "".join(parts)
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))

def to_universal(text: str, lang: str) -> str:
    if not text:
        return ""
    conv = _get_converter(lang)
    if conv is not None:
        try:
            out = conv.convert(text, sep="", lower=True)
        except Exception:
            out = _fallback_text(text)
    else:
        out = _fallback_text(text)
    out = _NON_AZ_SPACE.sub(" ", out.lower())
    return _MULTISPACE.sub(" ", out).strip()

def compute_cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    r_len, h_len = len(ref), len(hyp)
    dp = [[0] * (h_len + 1) for _ in range(r_len + 1)]
    for i in range(r_len + 1): dp[i][0] = i
    for j in range(h_len + 1): dp[0][j] = j
    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return min(1.0, dp[r_len][h_len] / r_len)

class AfricaWhisperJudge:
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        logger.info(f"Loading Africa-Whisper judge on {device}...")
        base = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v3-turbo",
            device_map=device,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
        )
        self.model = PeftModel.from_pretrained(
            base, "AfriSpeech/africa-whisper-checkpoints", subfolder="best_cer"
        )
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")

    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        if sr != 16000:
            audio_t = torchaudio.functional.resample(
                torch.from_numpy(audio)[None], orig_freq=sr, new_freq=16000
            )[0].numpy()
        else:
            audio_t = audio
        inputs = self.processor(
            audio_t.flatten(), sampling_rate=16000, return_tensors="pt"
        ).to(self.device, torch.float16 if "cuda" in self.device else torch.float32)
        with torch.no_grad():
            pred_ids = self.model.generate(input_features=inputs.input_features)
        raw_text = self.processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
        return _MULTISPACE.sub(" ", _NON_AZ_SPACE.sub(" ", raw_text.lower())).strip()

def collect_dev_sentences(max_per_lang: int = 2) -> Dict[str, List[str]]:
    dev_files = sorted(glob.glob("/mnt/volume_d2wey28/data/africa-tts/tokens/dev/txts/shard-*.jsonl"))
    by_lang = defaultdict(list)
    for f in dev_files:
        with open(f, "r", encoding="utf-8") as fp:
            for line in fp:
                d = json.loads(line)
                lang = d.get("iso") or d.get("language_id") or d.get("g2p")
                txt = d.get("text", "").strip()
                if lang and txt and len(txt) > 15 and len(txt) < 120:
                    if len(by_lang[lang]) < max_per_lang:
                        by_lang[lang].append(txt)
    return by_lang

def main():
    device = "cuda:0"
    out_dir = Path("/mnt/volume_d2wey28/projects/omnivoice/recipes/africa")
    out_dir.mkdir(parents=True, exist_ok=True)
    routing_file = out_dir / "language_text_routing.json"
    results_file = out_dir / "language_routing_eval_results.json"

    judge = AfricaWhisperJudge(device=device)

    logger.info("Loading Normal Text OmniVoice Model...")
    normal_model = OmniVoice.from_pretrained(
        "/mnt/volume_d2wey28/checkpoints/omnivoice_african_normal_text/checkpoint-best",
        device_map=device,
        dtype=torch.bfloat16,
    )

    logger.info("Loading Universal Text OmniVoice Model...")
    univ_model = OmniVoice.from_pretrained(
        "/mnt/volume_d2wey28/checkpoints/omnivoice_african_universal_text/checkpoint-best",
        device_map=device,
        dtype=torch.bfloat16,
    )

    logger.info("Collecting test sentences from dev shards...")
    dev_data = collect_dev_sentences(max_per_lang=2)
    logger.info(f"Collected sentences for {len(dev_data)} African languages.")

    routing_map = {}
    detailed_results = []

    languages = sorted(dev_data.keys())
    print("\n" + "="*80)
    print(f"{'Lang':<10} | {'Normal CER':<12} | {'Univ CER':<12} | {'Best Mode':<12} | {'Difference':<10}")
    print("-" * 80)

    for lang in languages:
        sentences = dev_data[lang]
        if not sentences:
            continue

        prompt_wav, prompt_text = get_prompt_voice(lang, preset="tts")

        normal_cers = []
        univ_cers = []

        for text in sentences:
            u_text = to_universal(text, lang)

            # 1. Synthesize with Normal Text Model
            try:
                a_norm = normal_model.generate(
                    text=text, language=lang, ref_audio=prompt_wav, ref_text=prompt_text, use_ipa=False, speed=0.85
                )[0]
                h_norm = judge.transcribe(a_norm, normal_model.sampling_rate)
                c_norm = compute_cer(u_text, h_norm)
            except Exception:
                c_norm = 1.0

            # 2. Synthesize with Universal Text Model
            try:
                a_univ = univ_model.generate(
                    text=u_text, language=lang, ref_audio=prompt_wav, ref_text=prompt_text, use_ipa=False, speed=0.85
                )[0]
                h_univ = judge.transcribe(a_univ, univ_model.sampling_rate)
                c_univ = compute_cer(u_text, h_univ)
            except Exception:
                c_univ = 1.0

            normal_cers.append(c_norm)
            univ_cers.append(c_univ)

        m_norm = float(np.mean(normal_cers))
        m_univ = float(np.mean(univ_cers))

        # Decision threshold: if universal is better by >= 1% CER, choose universal; else prefer normal orthography
        if m_univ < m_norm - 0.01:
            decision = "universal"
        else:
            decision = "normal"

        diff = (m_norm - m_univ) * 100
        routing_map[lang] = decision

        detailed_results.append({
            "language": lang,
            "normal_cer": round(m_norm * 100, 1),
            "universal_cer": round(m_univ * 100, 1),
            "best_mode": decision,
            "gain_with_best": round(abs(diff), 1),
        })

        diff_str = f"+{diff:.1f}%" if diff > 0 else f"{diff:.1f}%"
        print(f"{lang:<10} | {m_norm*100:<11.1f}% | {m_univ*100:<11.1f}% | {decision:<12} | {diff_str:<10}")

    print("="*80)

    # Save routing map and detailed results
    with open(routing_file, "w", encoding="utf-8") as f:
        json.dump(routing_map, f, indent=2)

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2)

    num_normal = sum(1 for v in routing_map.values() if v == "normal")
    num_univ = sum(1 for v in routing_map.values() if v == "universal")
    print(f"\nFinal Routing Summary: {num_normal} languages -> Normal Text, {num_univ} languages -> Universal Form.")
    print(f"Optimal routing saved to {routing_file}")

if __name__ == "__main__":
    main()
