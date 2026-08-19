"""Tests for the offline tiny-CPU smoke pieces.

Covers the offline language-detection heuristic (no fastText download) and a
short overfit check on the tiny GRU seq2seq, so the smoke path is guarded by CI
without pulling any heavy dependency.
"""

import importlib.util
import os
import random

import torch

from src.api import lid


def _load_smoke():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "scripts", "smoke.py")
    spec = importlib.util.spec_from_file_location("mnmt_smoke", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_offline_lid_scripts():
    # Force the offline heuristic; reset the cached detector.
    os.environ["MNMT_OFFLINE_LID"] = "1"
    lid._DETECTOR = None
    cases = {
        "Hello, world.": "en",
        "Bonjour le monde, comment allez vous": "fr",
        "Hola mundo, ¿como estas?": "es",
        "Guten Morgen, wie geht es dir?": "de",
        "नमस्ते दुनिया": "hi",
        "こんにちは世界": "ja",
        "Привет мир": "ru",
    }
    for text, expect in cases.items():
        assert lid.detect_lang(text) == expect, text
    assert lid.detect_lang("") is None


def test_tiny_seq2seq_overfits():
    mod = _load_smoke()
    torch.manual_seed(0)
    rng = random.Random(0)
    src, tgt = mod.make_dataset(128, rng)
    model = mod.Seq2Seq()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=mod.PAD)
    tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
    first = last = None
    for _ in range(120):
        logits = model(src, tgt_in)
        loss = loss_fn(logits.reshape(-1, mod.VOCAB), tgt_out.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
        last = loss.item()
    assert last < first * 0.5, (first, last)
