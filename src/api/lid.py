"""Language identification using fasttext-langdetect."""

from __future__ import annotations

from typing import Optional


_DETECTOR = None


def _detector():
    global _DETECTOR
    if _DETECTOR is None:
        from ftlangdetect import detect  # noqa: F401
        _DETECTOR = detect
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
