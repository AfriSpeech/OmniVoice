#!/usr/bin/env python3
"""
Benchmark comparison script: Original OmniVoice vs. OmniVoice-Africa.

Measures Character Error Rate (CER) on speech synthesized across native African
languages using AfriSpeech/africa-whisper-checkpoints as the neutral evaluation judge.

The CER is computed on the universal phonetic representation (a-z) produced by
Africa-Whisper via Africa-G2P universalization.

Both models are evaluated:
  - Conditioned on the EXACT SAME reference voice audio prompt
  - At the identical calibrated speaking rate (speed=0.85)

Generates:
  1. Detailed evaluation table (JSON & text summary)
  2. Publication-quality comparison bar chart ('benchmark_comparison.png')
"""

import argparse
import json
import logging
import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
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

# Native African language benchmark prompts (English excluded)
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
    """Converts transcript into universal lowercase Latin text (a-z) with single spaces."""
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
    """Compute character error rate (CER) using Levenshtein distance."""
    if not ref:
        return 0.0 if not hyp else 1.0

    r_len = len(ref)
    h_len = len(hyp)
    dp = [[0] * (h_len + 1) for _ in range(r_len + 1)]

    for i in range(r_len + 1):
        dp[i][0] = i
    for j in range(h_len + 1):
        dp[0][j] = j

    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )

    return min(1.0, dp[r_len][h_len] / r_len)


class AfricaWhisperJudge:
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        logger.info(f"Loading Africa-Whisper judge from AfriSpeech/africa-whisper-checkpoints (best_cer) on {device}...")
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
        # Clean to universal a-z
        return _MULTISPACE.sub(" ", _NON_AZ_SPACE.sub(" ", raw_text.lower())).strip()


def run_benchmark(
    base_model_path: str,
    africa_model_path: str,
    output_dir: str,
    device: str = "cuda:0",
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    audio_dir = out_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    judge = AfricaWhisperJudge(device=device)

    logger.info(f"Loading Base OmniVoice ({base_model_path}) on {device}...")
    base_model = OmniVoice.from_pretrained(
        base_model_path,
        device_map=device,
        dtype=torch.bfloat16 if "cuda" in device else torch.float32,
    )

    logger.info(f"Loading OmniVoice-Africa ({africa_model_path}) on {device}...")
    africa_model = OmniVoice.from_pretrained(
        africa_model_path,
        device_map=device,
        dtype=torch.bfloat16 if "cuda" in device else torch.float32,
    )

    results = []

    for item in BENCHMARK_PROMPTS:
        lang_name = item["language"]
        iso = item["iso"]
        g2p_lang = item["g2p_lang"]
        sentences = item["sentences"]

        prompt_wav, prompt_text = get_prompt_voice(iso, preset="tts")
        ref_clip_name = Path(prompt_wav).name if prompt_wav else "None"
        logger.info(f"=== Benchmarking {lang_name} ({iso}) [Ref Audio: {ref_clip_name}] ===")

        base_cers = []
        africa_cers = []

        for idx, text in enumerate(sentences):
            # 1. Generate with Base OmniVoice (speed=0.85, same reference audio, without IPA)
            base_audio = base_model.generate(
                text=text,
                language=iso,
                ref_audio=prompt_wav,
                ref_text=prompt_text,
                use_ipa=False,
                speed=0.85,
            )[0]
            sf.write(audio_dir / f"base_{iso}_{idx}.wav", base_audio, base_model.sampling_rate)

            # 2. Generate with OmniVoice-Africa (speed=0.85, same reference audio, with IPA)
            africa_audio = africa_model.generate(
                text=text,
                language=iso,
                ref_audio=prompt_wav,
                ref_text=prompt_text,
                use_ipa=True,
                speed=0.85,
            )[0]
            sf.write(audio_dir / f"africa_{iso}_{idx}.wav", africa_audio, africa_model.sampling_rate)

            # 3. Transcribe both with Africa-Whisper
            base_hyp = judge.transcribe(base_audio, base_model.sampling_rate)
            africa_hyp = judge.transcribe(africa_audio, africa_model.sampling_rate)

            # 4. Universalize reference text to match universal output of Africa-Whisper
            univ_ref = universalize_text(text, g2p_lang)

            # 5. Compute CER on universal phonetic text
            c_base = compute_cer(univ_ref, base_hyp)
            c_africa = compute_cer(univ_ref, africa_hyp)

            base_cers.append(c_base)
            africa_cers.append(c_africa)

            logger.info(f"  Sentence {idx + 1}:")
            logger.info(f"    Univ Ref:         {univ_ref}")
            logger.info(f"    Base Hyp (CER):   {base_hyp} ({c_base * 100:.1f}%)")
            logger.info(f"    Africa Hyp (CER): {africa_hyp} ({c_africa * 100:.1f}%)")

        mean_base = float(np.mean(base_cers))
        mean_africa = float(np.mean(africa_cers))
        rel_improvement = (mean_base - mean_africa) / mean_base * 100 if mean_base > 0 else 0.0

        results.append({
            "language": lang_name,
            "iso": iso,
            "base_cer": round(mean_base * 100, 1),
            "africa_cer": round(mean_africa * 100, 1),
            "reduction_pct": round(rel_improvement, 1),
        })

    # Save JSON summary
    summary_path = out_path / "benchmark_results.json"
    summary_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Results saved to {summary_path}")

    # Generate Publication Chart
    generate_chart(results, str(out_path / "benchmark_comparison.png"))
    return results


def generate_chart(results: List[Dict], chart_path: str):
    """Generate high-resolution comparative bar chart."""
    langs = [r["language"] for r in results]
    base_scores = [r["base_cer"] for r in results]
    africa_scores = [r["africa_cer"] for r in results]

    x = np.arange(len(langs))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.5, 6), dpi=300)
    fig.patch.set_facecolor("#0b0f19")
    ax.set_facecolor("#0f172a")

    rects1 = ax.bar(x - width / 2, base_scores, width, label="Original OmniVoice", color="#64748b", edgecolor="#475569")
    rects2 = ax.bar(x + width / 2, africa_scores, width, label="OmniVoice-Africa (Ours)", color="#0284c7", edgecolor="#38bdf8")

    # Add values on top of bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color="#cbd5e1", fontweight="bold")

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color="#38bdf8", fontweight="bold")

    # Percentage reduction tags
    for i, r in enumerate(results):
        red = r["reduction_pct"]
        if red > 0:
            ax.text(x[i] + width / 2, -3.5, f"-{red:.0f}%", ha="center", va="top",
                    fontsize=8, color="#34d399", fontweight="bold")

    ax.set_ylabel("Character Error Rate (CER %) — Lower is Better", fontsize=11, color="#f1f5f9", fontweight="bold")
    ax.set_title("OmniVoice vs. OmniVoice-Africa (Intelligibility Benchmark)\nEvaluated via Africa-Whisper Universal CER across African Languages (Same Voice Reference)",
                 fontsize=12.5, color="#f8fafc", fontweight="bold", pad=16)
    ax.set_xticks(x)
    ax.set_xticklabels(langs, rotation=15, ha="right", fontsize=10, color="#e2e8f0", fontweight="600")
    ax.tick_params(colors="#94a3b8")
    ax.grid(axis="y", linestyle="--", alpha=0.15, color="#cbd5e1")
    ax.set_ylim(-5, max(max(base_scores), max(africa_scores)) + 8)

    # Legend
    legend = ax.legend(frameon=True, facecolor="#1e293b", edgecolor="#334155", fontsize=9.5, loc="upper right")
    for text in legend.get_texts():
        text.set_color("#f1f5f9")

    plt.tight_layout()
    plt.savefig(chart_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    logger.info(f"Benchmark chart saved to {chart_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", type=str, default="/mnt/volume_d2wey28/models/OmniVoice-base")
    parser.add_argument("--africa_model", type=str, default="/mnt/volume_d2wey28/models/OmniVoice-africa-warmstart")
    parser.add_argument("--output_dir", type=str, default="benchmark_output")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    run_benchmark(
        base_model_path=args.base_model,
        africa_model_path=args.africa_model,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
