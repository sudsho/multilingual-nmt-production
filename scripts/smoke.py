"""Offline tiny-CPU smoke for multilingual-nmt-production.

The headline system fine-tunes mBART-50 (a ~2.4GB download) on a GPU and reports
spBLEU/chrF. None of that runs on a laptop with no network. This smoke proves the
*shape* of the pipeline end to end on CPU in a few seconds, with no downloads and
no pretrained weights:

  1. Build a tiny SYNTHETIC parallel corpus from a deterministic src->tgt
     transformation (target = reversed source over a tiny symbol vocab). This is
     a stand-in "translation" task that a tiny model can actually learn.
  2. Train a small GRU encoder-decoder WITH attention for a few hundred steps and
     show the training loss go down.
  3. Greedy-decode a held-out example and check the model reproduces the rule.
  4. Run the real language-detection component (src/api/lid.py) on sample strings
     in several scripts, using its offline heuristic fallback (no fastText model
     download).
  5. Confirm the mBART/transformers download path is guarded and skipped.

Run:  python scripts/smoke.py     (or: make smoke)

This is a SMOKE, not the real model. Real spBLEU needs mBART-50 + a GPU + a real
parallel corpus; see the README.
"""

from __future__ import annotations

import os
import random
import sys

import torch
import torch.nn as nn

# Print unicode (Devanagari, Kana, Cyrillic) even on a cp1252 Windows console.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Force the language detector onto its offline heuristic (no fastText download).
os.environ.setdefault("MNMT_OFFLINE_LID", "1")

# Make src importable when run as `python scripts/smoke.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.lid import detect_lang  # noqa: E402
from src.utils import set_seed  # noqa: E402


# ---- tiny vocab -----------------------------------------------------------
PAD, BOS, EOS = 0, 1, 2
SYMBOLS = list("abcdef")  # 6 content symbols
SYM2ID = {s: i + 3 for i, s in enumerate(SYMBOLS)}
ID2SYM = {i: s for s, i in SYM2ID.items()}
VOCAB = 3 + len(SYMBOLS)
SEQ_LEN = 5


def make_pair(rng: random.Random):
    """Deterministic toy 'translation': target is the reversed source."""
    src = [rng.choice(SYMBOLS) for _ in range(SEQ_LEN)]
    tgt = list(reversed(src))
    return src, tgt


def encode_src(sym_list):
    return [SYM2ID[s] for s in sym_list] + [EOS]


def encode_tgt(sym_list):
    return [BOS] + [SYM2ID[s] for s in sym_list] + [EOS]


def make_dataset(n, rng):
    src_batch, tgt_batch = [], []
    for _ in range(n):
        s, t = make_pair(rng)
        src_batch.append(torch.tensor(encode_src(s)))
        tgt_batch.append(torch.tensor(encode_tgt(t)))
    return torch.stack(src_batch), torch.stack(tgt_batch)


# ---- tiny attention seq2seq ----------------------------------------------
class Seq2Seq(nn.Module):
    def __init__(self, vocab=VOCAB, emb=32, hid=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=PAD)
        self.enc = nn.GRU(emb, hid, batch_first=True, bidirectional=True)
        self.bridge = nn.Linear(2 * hid, hid)
        self.dec = nn.GRU(emb, hid, batch_first=True)
        self.attn = nn.Linear(hid + 2 * hid, 1)
        self.out = nn.Linear(hid + 2 * hid, vocab)
        self.hid = hid

    def encode(self, src):
        e = self.emb(src)
        enc_out, h = self.enc(e)  # enc_out: (B,S,2H)
        h = torch.tanh(self.bridge(torch.cat([h[0], h[1]], dim=-1))).unsqueeze(0)
        return enc_out, h

    def step(self, tok, h, enc_out):
        e = self.emb(tok)  # (B,1,E)
        dec_out, h = self.dec(e, h)  # dec_out: (B,1,H)
        q = dec_out.expand(-1, enc_out.size(1), -1)  # (B,S,H)
        scores = self.attn(torch.cat([q, enc_out], dim=-1)).squeeze(-1)  # (B,S)
        w = torch.softmax(scores, dim=-1).unsqueeze(1)  # (B,1,S)
        ctx = torch.bmm(w, enc_out)  # (B,1,2H)
        logits = self.out(torch.cat([dec_out, ctx], dim=-1)).squeeze(1)  # (B,V)
        return logits, h

    def forward(self, src, tgt_in):
        enc_out, h = self.encode(src)
        logits = []
        for t in range(tgt_in.size(1)):
            step_logits, h = self.step(tgt_in[:, t:t + 1], h, enc_out)
            logits.append(step_logits)
        return torch.stack(logits, dim=1)  # (B,T,V)

    @torch.no_grad()
    def greedy(self, src, max_len=SEQ_LEN + 1):
        enc_out, h = self.encode(src)
        tok = torch.full((src.size(0), 1), BOS, dtype=torch.long)
        outs = []
        for _ in range(max_len):
            logits, h = self.step(tok, h, enc_out)
            tok = logits.argmax(-1, keepdim=True)
            nxt = tok.item()
            if nxt == EOS:
                break
            outs.append(nxt)
        return outs


def decode_ids(ids):
    return "".join(ID2SYM.get(i, "?") for i in ids)


def main():
    set_seed(7)
    rng = random.Random(7)

    train_src, train_tgt = make_dataset(2000, rng)
    model = Seq2Seq()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)

    print("=" * 60)
    print("multilingual-nmt-production  tiny-CPU offline smoke")
    print("=" * 60)
    print(f"synthetic task : reverse a length-{SEQ_LEN} sequence over {SYMBOLS}")
    print(f"model          : GRU enc-dec + attention, vocab={VOCAB}, "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    print(f"train pairs    : {train_src.size(0)}  (fully synthetic, no download)")
    print("-" * 60)

    tgt_in = train_tgt[:, :-1]
    tgt_out = train_tgt[:, 1:]
    bs = 64
    n = train_src.size(0)
    first_loss = None
    last_loss = None
    steps = 700
    for step in range(steps):
        idx = torch.randint(0, n, (bs,))
        logits = model(train_src[idx], tgt_in[idx])
        loss = loss_fn(logits.reshape(-1, VOCAB), tgt_out[idx].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first_loss is None:
            first_loss = loss.item()
        last_loss = loss.item()
        if step % 50 == 0 or step == steps - 1:
            print(f"step {step:4d}  loss {loss.item():.4f}")

    print("-" * 60)
    print(f"loss: {first_loss:.4f} -> {last_loss:.4f}  "
          f"({'decreased' if last_loss < first_loss else 'DID NOT DECREASE'})")

    # Greedy-decode a held-out example (fresh rng stream, unseen).
    hold_rng = random.Random(999)
    correct = 0
    trials = 8
    print("-" * 60)
    print("held-out greedy decode (src -> expected | got):")
    for _ in range(trials):
        s, t = make_pair(hold_rng)
        src = torch.tensor([encode_src(s)])
        got = decode_ids(model.greedy(src))
        exp = "".join(t)
        ok = got == exp
        correct += ok
        print(f"  {''.join(s)} -> {exp} | {got}  {'OK' if ok else 'x'}")
    acc = correct / trials

    # Language detection component (offline heuristic fallback).
    print("-" * 60)
    print("language detection (offline heuristic, no fastText download):")
    samples = [
        ("Hello, world.", "en"),
        ("Bonjour le monde, comment allez vous", "fr"),
        ("Hola mundo, ¿como estas?", "es"),
        ("Guten Morgen, wie geht es dir?", "de"),
        ("नमस्ते दुनिया", "hi"),
        ("こんにちは世界", "ja"),
        ("Привет мир", "ru"),
    ]
    lid_hits = 0
    for text, expect in samples:
        got = detect_lang(text)
        ok = got == expect
        lid_hits += ok
        print(f"  {text!r:45s} -> {got}  (expected {expect})  {'OK' if ok else 'x'}")
    lid_acc = lid_hits / len(samples)

    # Guard: confirm the mBART / transformers heavy path is skipped.
    print("-" * 60)
    if os.environ.get("MNMT_RUN_REAL") == "1":
        print("MNMT_RUN_REAL=1 set: real mBART-50 path would run (needs download + GPU).")
    else:
        print("mBART-50 / transformers download guarded and SKIPPED "
              "(set MNMT_RUN_REAL=1 to opt in).")

    # Verdict.
    print("=" * 60)
    ok_loss = last_loss < first_loss * 0.5
    ok_decode = acc >= 0.75
    ok_lid = lid_acc >= 0.85
    print(f"loss decreased >=2x : {ok_loss}  ({first_loss:.3f} -> {last_loss:.3f})")
    print(f"held-out decode acc : {acc:.2f}  (>=0.75 required)")
    print(f"language-id accuracy: {lid_acc:.2f}  (>=0.85 required)")
    if ok_loss and ok_decode and ok_lid:
        print("SMOKE OK")
        return 0
    print("SMOKE FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
