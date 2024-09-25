"""TranslationService: encapsulates model state + batched translate.

Separated from the FastAPI app so it can be reused by streamlit and CLI.
"""

from __future__ import annotations

import os
from typing import List, Optional


class TranslationService:
    def __init__(self):
        self._model = None
        self._tok = None

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self, ckpt: Optional[str] = None) -> None:
        from src.model import load_model_and_tokenizer
        ckpt = ckpt or os.environ.get("MNMT_CHECKPOINT", "facebook/mbart-large-50-many-to-many-mmt")
        self._model, self._tok = load_model_and_tokenizer(ckpt)

    def translate_batch(self, texts: List[str], src_codes: List[str], tgt_code: str,
                        num_beams: int = 5, length_penalty: float = 1.0) -> List[str]:
        from src.translate import translate
        if not self.is_loaded():
            self.load()
        out: List[Optional[str]] = [None] * len(texts)
        groups: dict[str, list[int]] = {}
        for i, c in enumerate(src_codes):
            groups.setdefault(c, []).append(i)
        for code, idxs in groups.items():
            chunk = [texts[i] for i in idxs]
            outs = translate(self._model, self._tok, chunk, code, tgt_code,
                             num_beams=num_beams, length_penalty=length_penalty)
            for j, i in enumerate(idxs):
                out[i] = outs[j]
        return [o for o in out if o is not None]
