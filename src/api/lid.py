"""Language identification.

Primary path: `fasttext-langdetect` (the `ftlangdetect` package). That library
lazily downloads a ~1GB-compressed fastText model on first use, which is not an
option on an offline box, so we guard it: set the environment variable
`MNMT_OFFLINE_LID=1` (or simply run somewhere `ftlangdetect` is not installed)
and detection falls back to a dependency-free script + char-ngram heuristic.

The heuristic is deliberately small. It nails non-Latin scripts by Unicode
range (Devanagari, Kana, Han, Hangul, Cyrillic, Arabic) and disambiguates the
common Latin languages (en/fr/es/de) with stopword and diacritic cues. It is
good enough for a smoke and for obvious inputs; the fastText model is what you
want for production accuracy on short, mixed, or noisy text.
"""

from __future__ import annotations

import os
from typing import Optional


_DETECTOR = None


# Small stopword cues for the common Latin-script languages.
_LATIN_CUES = {
    "en": {"the", "and", "is", "are", "you", "how", "hello", "world", "good", "of", "to", "a"},
    "fr": {"le", "la", "les", "bonjour", "monde", "est", "vous", "comment", "et", "un", "une"},
    "es": {"hola", "mundo", "como", "como", "estas", "el", "la", "gracias", "buenos", "dias", "y"},
    "de": {"guten", "morgen", "wie", "geht", "der", "die", "und", "ist", "dir", "hallo", "welt"},
}


def _heuristic_detect(text: str, low_memory: bool = True) -> dict:
    """Offline fallback with the same return shape as ftlangdetect.detect."""
    for ch in text:
        o = ord(ch)
        if 0x0900 <= o <= 0x097F:
            return {"lang": "hi", "score": 0.60}
        if 0x3040 <= o <= 0x30FF:  # Hiragana + Katakana
            return {"lang": "ja", "score": 0.60}
        if 0xAC00 <= o <= 0xD7A3:  # Hangul
            return {"lang": "ko", "score": 0.60}
        if 0x0400 <= o <= 0x04FF:  # Cyrillic
            return {"lang": "ru", "score": 0.60}
        if 0x0600 <= o <= 0x06FF:  # Arabic
            return {"lang": "ar", "score": 0.60}
        if 0x4E00 <= o <= 0x9FFF:  # Han (no kana seen first)
            return {"lang": "zh", "score": 0.55}

    lowered = text.lower()
    # Diacritic / punctuation cues first (strong signals).
    if any(c in lowered for c in "¿¡ñ"):
        return {"lang": "es", "score": 0.55}
    if "ß" in lowered or any(c in lowered for c in "äöü") and any(w in lowered for w in _LATIN_CUES["de"]):
        return {"lang": "de", "score": 0.55}

    tokens = [t.strip(".,!?;:\"'()") for t in lowered.split()]
    scores = {lang: sum(1 for t in tokens if t in cues) for lang, cues in _LATIN_CUES.items()}
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return {"lang": "en", "score": 0.30}  # default for unknown Latin text
    return {"lang": best, "score": 0.50}


def _detector():
    global _DETECTOR
    if _DETECTOR is None:
        if os.environ.get("MNMT_OFFLINE_LID") == "1":
            _DETECTOR = _heuristic_detect
        else:
            try:
                from ftlangdetect import detect
                _DETECTOR = detect
            except Exception:
                _DETECTOR = _heuristic_detect
    return _DETECTOR


def detect_lang(text: str) -> Optional[str]:
    """Return ISO 639-1 code for the input text, or None on failure."""
    text = (text or "").strip()
    if not text:
        return None
    fn = _detector()
    try:
        result = fn(text=text.replace("\n", " "), low_memory=True)
        return result.get("lang")
    except Exception:
        return None
