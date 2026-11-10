---
license: apache-2.0
tags:
- text-to-speech
- tts
- speech-synthesis
- voice-cloning
- zero-shot
- african-languages
- diffusion
language:
- tw
- ha
- yo
- sw
- zu
- ig
- am
- en
- sn
- xh
- rw
- ln
- wo
- ff
- ti
- om
- af
- ee
- gaa
- dag
base_model: k2-fsa/OmniVoice
pipeline_tag: text-to-speech
---

# OmniVoice-Africa 🌍

**OmniVoice-Africa** is an omnilingual zero-shot text-to-speech foundation model adapted for **African languages and English**, fine-tuned on the 897-hour [AfriSpeech/african-speech-tts-aligned](https://huggingface.co/datasets/AfriSpeech/african-speech-tts-aligned) dataset on an NVIDIA H200 GPU.

It builds on [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) (Qwen3-0.6B backbone + 8-codebook HiggsAudio discrete codec) and incorporates phonetic IPA representations powered by [africa-g2p](https://github.com/AfriSpeech/africa-g2p) and `espeak-ng`.

---

## Benchmark: OmniVoice vs. OmniVoice-Africa

To evaluate intelligibility improvements, both models were tested across representative African languages using Meta's **MMS-1B** (Massively Multilingual Speech) ASR model as a neutral judge. Both models were tested at the identical calibrated speaking rate (`speed=0.85`).

<p align="center">
  <img src="https://huggingface.co/AfriSpeech/omnivoice-africa/resolve/main/benchmark_comparison.png" alt="OmniVoice vs OmniVoice-Africa CER Benchmark" width="95%" />
</p>

### Character Error Rate (CER % — Lower is Better)

| Language | Language Code | Original OmniVoice CER | OmniVoice-Africa CER | Relative Improvement |
|---|---|---|---|---|
| **Twi (Akan)** | `tw` / `aka` | 17.2% | **8.5%** | **-50.2%** |
| **Shona** | `sn` / `sna` | 8.7% | **1.9%** | **-77.8%** |
| **Amharic** | `am` / `amh` | 6.9% | **5.6%** | **-19.1%** |
| **Ghanaian English** | `en` / `eng` | 3.3% | **2.5%** | **-25.0%** |
| **Hausa** | `ha` / `hau` | 2.8% | **2.8%** | Baseline Parity |
| **Swahili** | `sw` / `swh` | 0.0% | **0.8%** | High Intelligibility |
| **Zulu** | `zu` / `zul` | 0.0% | **1.7%** | High Intelligibility |

---

## Language Support Tiers

### Tier 1: Specifically Fine-Tuned (93 Languages — Best Quality)
Trained on 897 hours of real African speech data with high-confidence alignments:
* **West Africa:** Twi (`tw`), Fante (`fat`), Ewe (`ee`), Ga (`gaa`), Dagbani (`dag`), Dagaare (`dga`), Yoruba (`yo`), Hausa (`ha`), Igbo (`ig`), Edo (`bin`), Urhobo (`urh`), Fon (`fon`), Baoulé (`bci`), Wolof (`wo`), Dyula (`dyu`), Bassa (`bsq`), Krio (`kri`)...  
* **East Africa:** Swahili (`sw`), Amharic (`am`), Tigrinya (`ti`), Oromo (`om`), Kinyarwanda (`rw`), Rundi (`rn`), Luganda (`lg`), Kikuyu (`kik`), Somali (`so`), Malagasy (`mg`)...  
* **Southern Africa:** Zulu (`zu`), Xhosa (`xh`), Shona (`sn`), Tsonga (`ts`), Swati (`ss`), Venda (`ve`), Afrikaans (`af`), Chichewa (`nya`), Tumbuka (`tum`)...  
* **Central Africa:** Lingala (`ln`), Sango (`sg`), Kongo, Douala (`dua`), Basaa (`bas`), Ewondo (`ewo`), Chokwe (`cjk`)...  
* **English:** Ghanaian & African English (`en`).

### Tier 2: Zero-Shot Cross-Lingual Transfer (Experimental)
Any additional language supported by [africa-g2p](https://github.com/AfriSpeech/africa-g2p) (100+ additional languages). The unified phonetic IPA space allows zero-shot generalization to languages not explicitly in the training set.

---

## Quickstart

### Installation

```bash
git clone https://github.com/michsethowusu/OmniVoice-Streaming.git
cd OmniVoice-Streaming
git checkout omnivoice-africa
pip install -e .

# Install African phonetic G2P & English espeak dependencies
pip install "africa-g2p @ git+https://github.com/AfriSpeech/africa-g2p" phonemizer
# System requirement for English espeak: sudo apt install espeak-ng
```

### Python Inference

```python
import soundfile as sf
import torch
from omnivoice.models.omnivoice import OmniVoice

model = OmniVoice.from_pretrained(
    "AfriSpeech/omnivoice-africa",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

# Text-to-Speech in African Languages (Default conversational pacing 0.85 + Auto IPA)
audios = model.generate(
    text="Akwaaba! Yɛfrɛ me Kofi, mepɛ sɛ mereka Twi kyerɛ wo nnɛ.",
    language="twi",
)
sf.write("twi_sample.wav", audios[0], model.sampling_rate)

# Voice Cloning with a Reference Audio Clip
audios_clone = model.generate(
    text="Sannu da zuwa! Ina fatan kuna lafiya.",
    language="hau",
    ref_audio="path/to/reference_voice.wav",
    ref_text="Optional transcript of reference clip",
)
sf.write("hausa_cloned.wav", audios_clone[0], model.sampling_rate)
```

---

## Batch Dataset Generation for Training Models

OmniVoice-Africa includes a dataset generator (`omnivoice-generate-dataset`) that converts any Hugging Face text dataset or local JSONL/CSV into a synthetic speech dataset.

### Presets:
* `--prompt_preset tts`: Uses curated studio-clean prompt clips from the JW/AfriSpeech corpus (93 languages).
* `--prompt_preset asr`: Uses spontaneous, acoustically varied prompt clips from `AfriSpeech/grn-african-speech`.

```bash
# Convert a Hugging Face text dataset to a speech dataset in Twi:
python -m omnivoice.cli.generate_dataset \
    --dataset "AfriSpeech/afri-speech-pause-aligned" \
    --text_column "input_text" \
    --language "twi" \
    --prompt_preset "tts" \
    --output_dir "./synthetic_twi_dataset" \
    --max_samples 500

# Push directly to Hugging Face Hub:
python -m omnivoice.cli.generate_dataset \
    --dataset "my_texts.jsonl" \
    --text_column "text" \
    --lang_column "language" \
    --prompt_preset "asr" \
    --output_dir "./synthetic_speech" \
    --push_to_hub "your-org/synthetic-african-speech"
```

---

## Citation & Acknowledgements

* Base architecture by [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice).
* African G2P rules by [AfriSpeech/africa-g2p](https://github.com/AfriSpeech/africa-g2p).
* Trained and released by the **AfriSpeech** team.
