"""Data loading for multilingual NMT.

We pull parallel sentence pairs from Tatoeba via HuggingFace `datasets`. The
top-level entry point is `load_pairs(src, tgt, split)`. The exact dataset
config is left flexible so we can plug in our own JSONL files later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import json


@dataclass
class Pair:
    src_text: str
    tgt_text: str
    src_lang: str
    tgt_lang: str


def _from_jsonl(path: Path) -> List[Pair]:
    out: List[Pair] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out.append(Pair(
                src_text=row["src_text"],
                tgt_text=row["tgt_text"],
                src_lang=row["src_lang"],
                tgt_lang=row["tgt_lang"],
            ))
    return out


def _from_tatoeba(src: str, tgt: str, split: str):
    # Lazy import so a bare `pytest` run doesn't need datasets installed.
    from datasets import load_dataset
    ds = load_dataset("tatoeba", lang1=src, lang2=tgt, split=split)
    pairs: List[Pair] = []
    for row in ds:
        pairs.append(Pair(
            src_text=row["translation"][src],
            tgt_text=row["translation"][tgt],
            src_lang=src,
            tgt_lang=tgt,
        ))
    return pairs


def load_pairs(src: str, tgt: str, split: str = "train",
               jsonl_path: Optional[Path] = None) -> Iterable[Pair]:
    if jsonl_path is not None:
        return _from_jsonl(Path(jsonl_path))
    return _from_tatoeba(src, tgt, split)
