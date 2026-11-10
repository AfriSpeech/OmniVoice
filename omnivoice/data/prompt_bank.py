"""Preset prompt audio bank for zero-shot voice cloning in OmniVoice-Africa.

Supports two curated preset banks:
1. 'tts': Studio-clean, high-confidence prompt clips from the JW/AfriSpeech-aligned corpus (93 languages).
2. 'asr': Spontaneous, background-diverse speech clips from AfriSpeech/grn-african-speech.
"""

import json
import os
from pathlib import Path
from typing import Optional, Tuple

PROMPT_ROOT = Path(__file__).parent.parent.parent / "prompts"


def _normalize_lang(lang: str) -> str:
    lang = lang.lower().strip()
    # Map common aliases
    mapping = {
        "twi": "twi", "tw": "twi", "akan": "twi",
        "hausa": "hau", "ha": "hau", "hau": "hau",
        "yoruba": "yor", "yo": "yor", "yor": "yor",
        "swahili": "swa", "sw": "swa", "swh": "swa",
        "zulu": "zul", "zu": "zul", "zul": "zul",
        "english": "eng", "en": "eng", "eng": "eng",
        "amharic": "amh", "am": "amh", "amh": "amh",
        "igbo": "ibo", "ig": "ibo", "ibo": "ibo",
    }
    return mapping.get(lang, lang)


def get_prompt_voice(
    language: str,
    preset: str = "tts",
    prompts_dir: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Get reference prompt audio file and transcript for a given language.

    Args:
        language: Language code or name (e.g. 'twi', 'hau', 'yor', 'swa', 'eng').
        preset: Preset collection, either 'tts' (clean studio) or 'asr' (spontaneous/varied).
        prompts_dir: Optional custom directory containing prompt audio files.

    Returns:
        (wav_path, ref_text) or (None, None) if not available.
    """
    root = Path(prompts_dir) if prompts_dir else PROMPT_ROOT / preset
    if not root.exists():
        # Fallback to H200 standard location if running remotely
        h200_root = Path(f"/mnt/volume_d2wey28/projects/omnivoice/prompts/{preset}")
        if h200_root.exists():
            root = h200_root

    code = _normalize_lang(language)
    wav_path = root / f"{code}.wav"

    if not wav_path.exists():
        # Try finding any fallback in that language family
        for f in root.glob("*.wav"):
            if f.stem.startswith(code[:2]):
                wav_path = f
                code = f.stem
                break

    if not wav_path.exists():
        return None, None

    # Load transcript if voices.json exists
    ref_text = None
    voices_json = root / "voices.json"
    if voices_json.exists():
        try:
            with open(voices_json, "r", encoding="utf-8") as vf:
                data = json.load(vf)
                if code in data:
                    ref_text = data[code].get("text")
        except Exception:
            pass

    return str(wav_path), ref_text
