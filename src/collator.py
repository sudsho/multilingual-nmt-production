"""Pad-on-the-fly collator for the seq2seq trainer.

We avoid the HF DataCollatorForSeq2Seq dependency to keep the deps slim and
to make the pad logic explicit.
"""

from __future__ import annotations

from typing import Dict, List


class PadCollator:
    def __init__(self, pad_id: int, label_pad_id: int = -100):
        self.pad_id = pad_id
        self.label_pad_id = label_pad_id

    def __call__(self, batch: List[Dict[str, list]]):
        import torch

        def pad(seqs, value):
            max_len = max(len(s) for s in seqs)
            return torch.tensor(
                [list(s) + [value] * (max_len - len(s)) for s in seqs],
                dtype=torch.long,
            )

        return {
            "input_ids": pad([b["input_ids"] for b in batch], self.pad_id),
            "attention_mask": pad([b["attention_mask"] for b in batch], 0),
            "labels": pad([b["labels"] for b in batch], self.label_pad_id),
        }
