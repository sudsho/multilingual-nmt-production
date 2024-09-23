import torch

from src.collator import PadCollator


def test_pads_to_longest():
    c = PadCollator(pad_id=0)
    out = c([
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [4, 5]},
        {"input_ids": [9, 8],    "attention_mask": [1, 1],    "labels": [7, 6, 5, 4]},
    ])
    assert out["input_ids"].shape == (2, 3)
    assert out["labels"].shape == (2, 4)
    # pad token is 0 on input ids, -100 on labels
    assert int(out["input_ids"][1, 2]) == 0
    assert int(out["labels"][0, 2]) == -100


def test_returns_long_tensors():
    c = PadCollator(pad_id=0)
    out = c([
        {"input_ids": [1], "attention_mask": [1], "labels": [2]},
    ])
    assert out["input_ids"].dtype == torch.long
    assert out["labels"].dtype == torch.long
