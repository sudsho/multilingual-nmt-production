"""Data loading for multilingual NMT.

We pull parallel sentence pairs from Tatoeba via HuggingFace `datasets`. The
top-level entry point is `load_pairs(src, tgt, split)`. The exact dataset
config is left flexible so we can plug in our own JSONL files later.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Pair:
    src_text: str
    tgt_text: str
    src_lang: str
    tgt_lang: str


def load_pairs(src: str, tgt: str, split: str = "train"):
    """Return an iterable of Pair objects.

    Stub for now. Real implementation will use `datasets.load_dataset`.
    """
    raise NotImplementedError
