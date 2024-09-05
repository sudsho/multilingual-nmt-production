"""Preprocessing for mBART-50.

The MBart50 tokenizer expects you to set `src_lang` before encoding the source
side and `tgt_lang` before encoding the target side. We wrap that so callers
don't have to remember the dance.
"""

from __future__ import annotations

from typing import Dict, List


# mBART-50 language codes
LANGS = {
    "en": "en_XX",
    "hi": "hi_IN",
    "fr": "fr_XX",
    "de": "de_DE",
    "es": "es_XX",
    "it": "it_IT",
    "ja": "ja_XX",
    "ko": "ko_KR",
    "zh": "zh_CN",
    "ar": "ar_AR",
    "ru": "ru_RU",
    "pt": "pt_XX",
}


def to_mbart_code(short: str) -> str:
    if short in LANGS:
        return LANGS[short]
    if short in LANGS.values():
        return short
    raise ValueError(f"unknown language: {short}")


def encode_pair(tokenizer, src_text: str, tgt_text: str, src_lang: str, tgt_lang: str,
                max_source_length: int = 128, max_target_length: int = 128) -> Dict[str, List[int]]:
    src_code = to_mbart_code(src_lang)
    tgt_code = to_mbart_code(tgt_lang)
    tokenizer.src_lang = src_code
    model_inputs = tokenizer(
        src_text,
        max_length=max_source_length,
        truncation=True,
    )
    tokenizer.tgt_lang = tgt_code
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            tgt_text,
            max_length=max_target_length,
            truncation=True,
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
