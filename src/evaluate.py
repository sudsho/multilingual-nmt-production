"""Evaluation: spBLEU + chrF via sacrebleu."""

from __future__ import annotations

from typing import List

import sacrebleu


def spbleu(hyps: List[str], refs: List[str]) -> float:
    """Sentencepiece BLEU using the FLORES tokenizer."""
    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="flores200")
    return bleu.score


def chrf(hyps: List[str], refs: List[str]) -> float:
    return sacrebleu.corpus_chrf(hyps, [refs]).score
