"""Model loader for mBART-50 many-to-many."""

from __future__ import annotations

from typing import Tuple

from transformers import MBart50TokenizerFast, MBartForConditionalGeneration


def load_model_and_tokenizer(name: str = "facebook/mbart-large-50-many-to-many-mmt") -> Tuple[MBartForConditionalGeneration, MBart50TokenizerFast]:
    tok = MBart50TokenizerFast.from_pretrained(name)
    model = MBartForConditionalGeneration.from_pretrained(name)
    return model, tok
