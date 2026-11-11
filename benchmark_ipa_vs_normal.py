#!/usr/bin/env python3
"""
Comparative benchmark: OmniVoice Base vs. OmniVoice-Africa (IPA) vs. OmniVoice-Africa (Normal Text).
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
        "sentences": [
            "Akwaaba! Yɛfrɛ me Kofi, mepɛ sɛ mereka Twi kyerɛ wo nnɛ.",
            "Ɔsoro ne asase bɛtwa mu na me nsɛm deɛ ɛrentwa mu da.",
        ],
    },
    {
        "language": "Hausa",
        "iso": "ha",
        "g2p_lang": "hau",
        "sentences": [
            "Sannu da zuwa! Ina fatan kuna lafiya.",
            "Ilimi haske ne mai shiryar da rayuwar dan adam zuwa ga hanya madaidaiciya.",
        ],
    },
    {
        "language": "Yoruba",
        "iso": "yo",
        "g2p_lang": "yor",
        "sentences": [
            "Bawo ni! Inu mi dun lati ba yin soro loni.",
            "Ile la ti n ko eso r'ode, eniyan to ba fe se rere gbodo bere lati ile.",
        ],
    },
    {
        "language": "Swahili",
        "iso": "sw",
        "g2p_lang": "swh",
        "sentences": [
            "Habari za asubuhi! Karibu sana katika safari yetu.",
            "Elimu ni ufunguo wa maisha na mwongozo bora wa mafanikio yetu.",
        ],
    },
    {
        "language": "Zulu",
        "iso": "zu",
        "g2p_lang": "zul",
        "sentences": [
            "Sawubona unjani namhlanje? Ngijabule kakhulu ukukubona.",
            "Umuntu ngumuntu ngabantu, masisebenzisane ndawonye ngokuthula.",
        ],
    },
    {
        "language": "Amharic",
        "iso": "am",
        "g2p_lang": "amh",
        "sentences": [
            "እንደምን አደራችሁ! ዛሬ ስላገኘኋችሁ በጣም ደስ ብሎኛል::",
            "ትምህርት የሕይወት መሠረት እና የተሻለ የወደፊት መንገድ ነው::",
        ],
    },
    {
        "language": "Shona",
        "iso": "sn",
        "g2p_lang": "sna",
        "sentences": [
            "Mhoroi, makadii henyu nhasi? Tinofara chaizvo kukuonai.",
            "Ruzivo isimba rinovhura mikana mitsva muupenyu hwedu.",
        ],
    },
    {
        "language": "Ghanaian English",
        "iso": "en",
        "g2p_lang": "eng",
        "sentences": [
            "Good morning! We are excited to present our new speech technology today.",
            "Education is the most powerful weapon which you can use to change the world.",
        ],
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
        logger.info(f"Loading Africa-Whisper judge from AfriSpeech/africa-whisper-checkpoints on {device}...")
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
    out_dir = Path("/mnt/volume_d2wey28/projects/omnivoice/benchmark_ipa_vs_normal_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    judge = AfricaWhisperJudge(device=device)

    logger.info("Loading Base OmniVoice...")
    base_model = OmniVoice.from_pretrained(
        "/mnt/volume_d2wey28/models/OmniVoice-base",
        device_map=device,
        dtype=torch.bfloat16,
    )

    logger.info("Loading OmniVoice-Africa (IPA)...")
    ipa_model = OmniVoice.from_pretrained(
        "/mnt/volume_d2wey28/models/OmniVoice-africa-warmstart",
        device_map=device,
        dtype=torch.bfloat16,
    )

    logger.info("Loading OmniVoice-Africa (Normal Text)...")
    normal_model = OmniVoice.from_pretrained(
        "/mnt/volume_d2wey28/checkpoints/omnivoice_african_normal_text/checkpoint-best",
        device_map=device,
        dtype=torch.bfloat16,
    )

    results = []

    for item in BENCHMARK_PROMPTS:
        lang_name = item["language"]
        iso = item["iso"]
        g2p_lang = item["g2p_lang"]
        sentences = item["sentences"]

        prompt_wav, prompt_text = get_prompt_voice(iso, preset="tts")
        logger.info(f"=== Benchmarking {lang_name} ({iso}) ===")

        base_cers = []
        ipa_cers = []
        normal_cers = []

        for idx, text in enumerate(sentences):
            # 1. Base OmniVoice (speed=0.85, without IPA)
            b_audio = base_model.generate(
                text=text, language=iso, ref_audio=prompt_wav, ref_text=prompt_text, use_ipa=False, speed=0.85
            )[0]
            sf.write(audio_dir / f"base_{iso}_{idx}.wav", b_audio, base_model.sampling_rate)

            # 2. OmniVoice Africa (IPA, speed=0.85)
            i_audio = ipa_model.generate(
                text=text, language=iso, ref_audio=prompt_wav, ref_text=prompt_text, use_ipa=True, speed=0.85
            )[0]
            sf.write(audio_dir / f"ipa_{iso}_{idx}.wav", i_audio, ipa_model.sampling_rate)

            # 3. OmniVoice Africa (Normal Text, speed=0.85)
            n_audio = normal_model.generate(
                text=text, language=iso, ref_audio=prompt_wav, ref_text=prompt_text, use_ipa=False, speed=0.85
            )[0]
            sf.write(audio_dir / f"normal_{iso}_{idx}.wav", n_audio, normal_model.sampling_rate)

            # Transcribe all three
            b_hyp = judge.transcribe(b_audio, base_model.sampling_rate)
            i_hyp = judge.transcribe(i_audio, ipa_model.sampling_rate)
            n_hyp = judge.transcribe(n_audio, normal_model.sampling_rate)

            univ_ref = universalize_text(text, g2p_lang)

            c_base = compute_cer(univ_ref, b_hyp)
            c_ipa = compute_cer(univ_ref, i_hyp)
            c_normal = compute_cer(univ_ref, n_hyp)

            base_cers.append(c_base)
            ipa_cers.append(c_ipa)
            normal_cers.append(c_normal)

            logger.info(f"  Sentence {idx + 1}:")
            logger.info(f"    Target:      {univ_ref}")
            logger.info(f"    Base CER:    {c_base*100:.1f}% -> {b_hyp}")
            logger.info(f"    IPA CER:     {c_ipa*100:.1f}% -> {i_hyp}")
            logger.info(f"    Normal CER:  {c_normal*100:.1f}% -> {n_hyp}")

        m_base = float(np.mean(base_cers))
        m_ipa = float(np.mean(ipa_cers))
        m_normal = float(np.mean(normal_cers))

        results.append({
            "language": lang_name,
            "iso": iso,
            "base_cer": round(m_base * 100, 1),
            "ipa_cer": round(m_ipa * 100, 1),
            "normal_cer": round(m_normal * 100, 1),
        })

    summary_path = out_dir / "benchmark_ipa_vs_normal_results.json"
    summary_path.write_text(json.dumps(results, indent=2))

    print("\n" + "="*75)
    print("OMNIVOICE BENCHMARK: BASE vs. AFRICA (IPA) vs. AFRICA (NORMAL TEXT)")
    print("="*75)
    print(f"{'Language':<18} | {'Base CER':<10} | {'Africa (IPA)':<14} | {'Africa (Normal Text)':<20}")
    print("-" * 75)
    for r in results:
        print(f"{r['language']:<18} | {r['base_cer']:<9.1f}% | {r['ipa_cer']:<13.1f}% | {r['normal_cer']:<19.1f}%")
    print("="*75)

if __name__ == "__main__":
    main()
