#!/usr/bin/env python3
"""
4-Way Comparative Benchmark:
1. Base OmniVoice (151k Native Tokenizer, Zero-Shot)
2. OmniVoice-Africa (4K IPA Tokenizer, Warm-Start, with IPA)
3. OmniVoice-Africa (151k Native Tokenizer, Full Finetune, Normal Text)
4. OmniVoice-Africa (4K Text Tokenizer text_sp4000, Warm-Start, Normal Text)

Evaluated via Africa-Whisper universal CER across native African languages.
"""

import json
import logging
import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

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

BENCHMARK_PROMPTS = [
    {
        "language": "Twi (Akan)",
        "iso": "tw",
        "g2p_lang": "twi",
        "text": "Akwaaba! Yɛfrɛ me Kofi, mepɛ sɛ mereka Twi kyerɛ wo nnɛ.",
    },
    {
        "language": "Hausa",
        "iso": "ha",
        "g2p_lang": "hau",
        "text": "Sannu da zuwa! Ina fatan kuna lafiya.",
    },
    {
        "language": "Yoruba",
        "iso": "yo",
        "g2p_lang": "yor",
        "text": "Bawo ni! Inu mi dun lati ba yin soro loni.",
    },
    {
        "language": "Swahili",
        "iso": "sw",
        "g2p_lang": "swh",
        "text": "Habari za asubuhi! Karibu sana katika safari yetu.",
    },
    {
        "language": "Zulu",
        "iso": "zu",
        "g2p_lang": "zul",
        "text": "Sawubona unjani namhlanje? Ngijabule kakhulu ukukubona.",
    },
    {
        "language": "Amharic",
        "iso": "am",
        "g2p_lang": "amh",
        "text": "እንደምን አደራችሁ! ዛሬ ስላገኘኋችሁ በጣም ደስ ብሎኛል::",
    },
    {
        "language": "Shona",
        "iso": "sn",
        "g2p_lang": "sna",
        "text": "Mhoroi, makadii henyu nhasi? Tinofara chaizvo kukuonai.",
    },
    {
        "language": "Ghanaian English",
        "iso": "en",
        "g2p_lang": "eng",
        "text": "Good morning! We are excited to present our new speech technology today.",
    },
]

_NON_AZ_SPACE = re.compile(r"[^a-z\s]+")
_MULTISPACE = re.compile(r"\s+")

RULE_ALIASES = {
    "tw": "twi", "aka": "twi", "akan": "twi",
    "sw": "swh", "swa": "swh",
    "ha": "hau",
    "yo": "yor",
    "ig": "ibo",
    "zu": "zul",
    "am": "amh",
    "sn": "sna",
    "en": "eng",
}

@lru_cache(maxsize=1024)
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

def universalize_text(text: str, lang: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    code = RULE_ALIASES.get(lang.lower(), lang.lower())
    conv = _get_converter(code) or _get_converter(lang.lower())
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

def main():
    device = "cuda:0"
    out_dir = Path("/mnt/volume_d2wey28/projects/omnivoice/benchmark_4way_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    judge = AfricaWhisperJudge(device=device)

    logger.info("1. Loading Base OmniVoice (151k Native, Zero-Shot)...")
    base_model = OmniVoice.from_pretrained(
        "/mnt/volume_d2wey28/models/OmniVoice-base",
        device_map=device,
        dtype=torch.bfloat16,
    )

    logger.info("2. Loading OmniVoice-Africa (4K IPA Tokenizer)...")
    ipa_model = OmniVoice.from_pretrained(
        "/mnt/volume_d2wey28/models/OmniVoice-africa-warmstart",
        device_map=device,
        dtype=torch.bfloat16,
    )

    logger.info("3. Loading OmniVoice-Africa (151k Native Tokenizer, Normal Text)...")
    native_text_model = OmniVoice.from_pretrained(
        "/mnt/volume_d2wey28/checkpoints/omnivoice_african_normal_text/checkpoint-best",
        device_map=device,
        dtype=torch.bfloat16,
    )

    logger.info("4. Loading OmniVoice-Africa (4K Text Tokenizer text_sp4000, Normal Text)...")
    sp4000_text_model = OmniVoice.from_pretrained(
        "/mnt/volume_d2wey28/checkpoints/omnivoice_african_text_sp4000/checkpoint-best",
        device_map=device,
        dtype=torch.bfloat16,
    )

    results = []

    for item in BENCHMARK_PROMPTS:
        lang_name = item["language"]
        iso = item["iso"]
        g2p_lang = item["g2p_lang"]
        text = item["text"]

        prompt_wav, prompt_text = get_prompt_voice(iso, preset="tts")
        logger.info(f"=== Evaluating {lang_name} ({iso}) ===")
        univ_ref = universalize_text(text, g2p_lang)

        # 1. Base OmniVoice
        a1 = base_model.generate(
            text=text, language=iso, ref_audio=prompt_wav, ref_text=prompt_text, use_ipa=False, speed=0.85
        )[0]
        sf.write(audio_dir / f"m1_base_{iso}.wav", a1, base_model.sampling_rate)
        h1 = judge.transcribe(a1, base_model.sampling_rate)
        c1 = compute_cer(univ_ref, h1)

        # 2. OmniVoice Africa (IPA, 4k tokenizer)
        a2 = ipa_model.generate(
            text=text, language=iso, ref_audio=prompt_wav, ref_text=prompt_text, use_ipa=True, speed=0.85
        )[0]
        sf.write(audio_dir / f"m2_ipa_{iso}.wav", a2, ipa_model.sampling_rate)
        h2 = judge.transcribe(a2, ipa_model.sampling_rate)
        c2 = compute_cer(univ_ref, h2)

        # 3. OmniVoice Africa (151k Native Tokenizer, Normal Text)
        a3 = native_text_model.generate(
            text=text, language=iso, ref_audio=prompt_wav, ref_text=prompt_text, use_ipa=False, speed=0.85
        )[0]
        sf.write(audio_dir / f"m3_native_text_{iso}.wav", a3, native_text_model.sampling_rate)
        h3 = judge.transcribe(a3, native_text_model.sampling_rate)
        c3 = compute_cer(univ_ref, h3)

        # 4. OmniVoice Africa (4K Text Tokenizer text_sp4000, Normal Text)
        a4 = sp4000_text_model.generate(
            text=text, language=iso, ref_audio=prompt_wav, ref_text=prompt_text, use_ipa=False, speed=0.85
        )[0]
        sf.write(audio_dir / f"m4_sp4000_text_{iso}.wav", a4, sp4000_text_model.sampling_rate)
        h4 = judge.transcribe(a4, sp4000_text_model.sampling_rate)
        c4 = compute_cer(univ_ref, h4)

        logger.info(f"  Target:     {univ_ref}")
        logger.info(f"  Base (151k):         CER={c1*100:.1f}% -> {h1}")
        logger.info(f"  Africa (4K IPA):     CER={c2*100:.1f}% -> {h2}")
        logger.info(f"  Africa (151k Text):  CER={c3*100:.1f}% -> {h3}")
        logger.info(f"  Africa (4K Text):    CER={c4*100:.1f}% -> {h4}")

        results.append({
            "language": lang_name,
            "iso": iso,
            "base_151k_cer": round(c1 * 100, 1),
            "africa_4k_ipa_cer": round(c2 * 100, 1),
            "africa_151k_text_cer": round(c3 * 100, 1),
            "africa_4k_text_cer": round(c4 * 100, 1),
        })

    summary_path = out_dir / "benchmark_4way_results.json"
    summary_path.write_text(json.dumps(results, indent=2))

    print("\n" + "="*95)
    print("4-WAY TOKENIZER & TEXT COMPARISON ON OMNIVOICE")
    print("="*95)
    print(f"{'Language':<18} | {'Base (151k)':<12} | {'4K IPA Tokenizer':<18} | {'151k Normal Text':<18} | {'4K Normal Text':<18}")
    print("-" * 95)
    for r in results:
        print(f"{r['language']:<18} | {r['base_151k_cer']:<11.1f}% | {r['africa_4k_ipa_cer']:<17.1f}% | {r['africa_151k_text_cer']:<17.1f}% | {r['africa_4k_text_cer']:<17.1f}%")
    print("="*95)

if __name__ == "__main__":
    main()
