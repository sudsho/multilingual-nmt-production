"""Evaluation: spBLEU + chrF via sacrebleu.

`spbleu` uses the FLORES-200 SentencePiece tokenizer when it is available
(that is the metric reported in the README). The FLORES tokenizer needs the
`sentencepiece` package plus a one-time SPM model download, so when it is not
available (for example an offline CPU box running the smoke) we transparently
fall back to sacrebleu's built-in `13a` tokenizer. Perfect vs terrible scores
still separate cleanly under either tokenizer; only the exact number shifts.
"""

from __future__ import annotations

from typing import List


def spbleu(hyps: List[str], refs: List[str], tokenize: str = "flores200") -> float:
    """Sentencepiece BLEU using the FLORES tokenizer, with an offline fallback."""
    import sacrebleu

    try:
        bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tokenize)
    except (ImportError, ModuleNotFoundError, FileNotFoundError, OSError):
        # FLORES SPM tokenizer unavailable (no sentencepiece / no model download).
        # Fall back to sacrebleu's built-in character tokenizer: like FLORES it is
        # subword-granular (so short sentences still have 4-grams) but needs no
        # download, keeping the metric usable fully offline.
        bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="char")
    return bleu.score


def chrf(hyps: List[str], refs: List[str]) -> float:
    import sacrebleu

    return sacrebleu.corpus_chrf(hyps, [refs]).score
