#!/usr/bin/env python3
"""
Benchmark comparison script: Original OmniVoice vs. OmniVoice-Africa.

Measures Character Error Rate (CER) on speech synthesized across representative
African languages using Meta MMS-1B ASR as the neutral evaluation judge.

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
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import AutoProcessor, Wav2Vec2ForCTC

from omnivoice.models.omnivoice import OmniVoice

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Test sentences across representative language families
BENCHMARK_PROMPTS = [
    {
        "language": "Twi (Akan)",
        "iso": "tw",
        "mms_lang": "aka",
        "sentences": [
            "Akwaaba! Yɛfrɛ me Kofi, mepɛ sɛ mereka Twi kyerɛ wo nnɛ.",
            "Ɔsoro ne asase bɛtwa mu na me nsɛm deɛ ɛrentwa mu da.",
        ],
    },
    {
        "language": "Hausa",
        "iso": "ha",
        "mms_lang": "hau",
        "sentences": [
            "Sannu da zuwa! Ina fatan kuna lafiya.",
            "Ilimi haske ne mai shiryar da rayuwar dan adam zuwa ga hanya madaidaiciya.",
        ],
    },
    {
        "language": "Yoruba",
        "iso": "yo",
        "mms_lang": "yor",
        "sentences": [
            "Bawo ni! Inu mi dun lati ba yin soro loni.",
            "Ile la ti n ko eso r'ode, eniyan to ba fe se rere gbodo bere lati ile.",
        ],
    },
    {
        "language": "Swahili",
        "iso": "sw",
        "mms_lang": "swh",
        "sentences": [
            "Habari za asubuhi! Karibu sana katika safari yetu.",
            "Elimu ni ufunguo wa maisha na mwongozo bora wa mafanikio yetu.",
        ],
    },
    {
        "language": "Zulu",
        "iso": "zu",
        "mms_lang": "zul",
        "sentences": [
            "Sawubona unjani namhlanje? Ngijabule kakhulu ukukubona.",
            "Umuntu ngumuntu ngabantu, masisebenzisane ndawonye ngokuthula.",
        ],
    },
    {
        "language": "Amharic",
        "iso": "am",
        "mms_lang": "amh",
        "sentences": [
            "እንደምን አደራችሁ! ዛሬ ስላገኘኋችሁ በጣም ደስ ብሎኛል::",
            "ትምህርት የሕይወት መሠረት እና የተሻለ የወደፊት መንገድ ነው::",
        ],
    },
    {
        "language": "Shona",
        "iso": "sn",
        "mms_lang": "sna",
        "sentences": [
            "Mhoroi, makadii henyu nhasi? Tinofara chaizvo kukuonai.",
            "Ruzivo isimba rinovhura mikana mitsva muupenyu hwedu.",
        ],
    },
    {
        "language": "Ghanaian English",
        "iso": "en",
        "mms_lang": "eng",
        "sentences": [
            "Hello Ghana, welcome to our AI speech synthesis showcase today!",
            "Technology gives us the opportunity to preserve and amplify our African heritage.",
        ],
    },
]


def normalize_for_cer(text: str) -> str:
    """Normalize text by lowercasing, stripping punctuation, and collapsing whitespace."""
    text = unicodedata.normalize("NFKC", text.lower())
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_cer(ref: str, hyp: str) -> float:
    """Compute character error rate (CER) using Levenshtein distance."""
    ref_norm = normalize_for_cer(ref)
    hyp_norm = normalize_for_cer(hyp)
    if not ref_norm:
        return 0.0 if not hyp_norm else 1.0

    r_len = len(ref_norm)
    h_len = len(hyp_norm)
    dp = [[0] * (h_len + 1) for _ in range(r_len + 1)]

    for i in range(r_len + 1):
        dp[i][0] = i
    for j in range(h_len + 1):
        dp[0][j] = j

    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            cost = 0 if ref_norm[i - 1] == hyp_norm[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )

    return min(1.0, dp[r_len][h_len] / r_len)


class ASRJudge:
    def __init__(self, device: str = "cpu"):
        self.device = device
        logger.info(f"Loading MMS-1B ASR judge model on {device}...")
        self.processor = AutoProcessor.from_pretrained("facebook/mms-1b-all")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all").to(device)

    def transcribe(self, audio: np.ndarray, sr: int, mms_lang: str) -> str:
        # Resample to 16kHz for MMS-1B
        if sr != 16000:
            audio_t = torchaudio.functional.resample(
                torch.from_numpy(audio)[None], orig_freq=sr, new_freq=16000
            )[0].numpy()
        else:
            audio_t = audio

        try:
            self.processor.tokenizer.set_target_lang(mms_lang)
            self.model.load_adapter(mms_lang)
        except Exception as e:
            logger.warning(f"Could not load MMS adapter for {mms_lang}: {e}")

        inputs = self.processor(audio_t, sampling_rate=16000, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)[0]
        return self.processor.decode(pred_ids)


def run_benchmark(
    base_model_path: str,
    africa_model_path: str,
    output_dir: str,
    device: str = "cpu",
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    audio_dir = out_path / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    judge = ASRJudge(device=device)

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
        mms_lang = item["mms_lang"]
        sentences = item["sentences"]

        base_cers = []
        africa_cers = []

        logger.info(f"=== Benchmarking {lang_name} ({iso}) ===")

        for idx, text in enumerate(sentences):
            # 1. Generate with Base OmniVoice (calibrated to same speed=0.85, without IPA)
            base_audio = base_model.generate(
                text=text,
                language=iso,
                use_ipa=False,
                speed=0.85,
            )[0]
            sf.write(audio_dir / f"base_{iso}_{idx}.wav", base_audio, base_model.sampling_rate)

            # 2. Generate with OmniVoice-Africa (speed=0.85, with IPA)
            africa_audio = africa_model.generate(
                text=text,
                language=iso,
                use_ipa=True,
                speed=0.85,
            )[0]
            sf.write(audio_dir / f"africa_{iso}_{idx}.wav", africa_audio, africa_model.sampling_rate)

            # 3. Transcribe both with MMS-1B
            base_hyp = judge.transcribe(base_audio, base_model.sampling_rate, mms_lang)
            africa_hyp = judge.transcribe(africa_audio, africa_model.sampling_rate, mms_lang)

            # 4. Compute CER
            c_base = compute_cer(text, base_hyp)
            c_africa = compute_cer(text, africa_hyp)

            base_cers.append(c_base)
            africa_cers.append(c_africa)

            logger.info(f"  Sentence {idx + 1}:")
            logger.info(f"    Ref:             {text}")
            logger.info(f"    Base Hyp:        {base_hyp} (CER: {c_base * 100:.1f}%)")
            logger.info(f"    OmniVoice-Af Hyp: {africa_hyp} (CER: {c_africa * 100:.1f}%)")

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

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
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
            ax.text(x[i] + width / 2, -4.5, f"-{red:.0f}%", ha="center", va="top",
                    fontsize=8, color="#34d399", fontweight="bold")

    ax.set_ylabel("Character Error Rate (CER %) — Lower is Better", fontsize=11, color="#f1f5f9", fontweight="bold")
    ax.set_title("OmniVoice vs. OmniVoice-Africa (Intelligibility Benchmark)\nEvaluated via Meta MMS-1B ASR across African Languages",
                 fontsize=13, color="#f8fafc", fontweight="bold", pad=16)
    ax.set_xticks(x)
    ax.set_xticklabels(langs, rotation=25, ha="right", fontsize=9, color="#e2e8f0", fontweight="600")
    ax.tick_params(colors="#94a3b8")
    ax.grid(axis="y", linestyle="--", alpha=0.15, color="#cbd5e1")
    ax.set_ylim(-6, max(max(base_scores), max(africa_scores)) + 8)

    # Legend
    legend = ax.legend(frameon=True, facecolor="#1e293b", edgecolor="#334155", fontsize=9, loc="upper right")
    for text in legend.get_texts():
        text.set_color("#f1f5f9")

    plt.tight_layout()
    plt.savefig(chart_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    logger.info(f"Benchmark chart saved to {chart_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", type=str, default="/mnt/volume_d2wey28/models/OmniVoice-base")
    parser.add_argument("--africa_model", type=str, default="/mnt/volume_d2wey28/models/OmniVoice-african-merged")
    parser.add_argument("--output_dir", type=str, default="benchmark_output")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_benchmark(
        base_model_path=args.base_model,
        africa_model_path=args.africa_model,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
