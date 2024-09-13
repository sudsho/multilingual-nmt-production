"""Build a parallel jsonl corpus from raw Tatoeba dumps.

Filters: drop pairs longer than `max_words` on either side, dedupe by src text,
optionally subsample with `--limit`.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--tgt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-words", type=int, default=60)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def main():
    args = parse_args()
    from datasets import load_dataset
    ds = load_dataset("tatoeba", lang1=args.src, lang2=args.tgt, split="train")
    seen = set()
    out_rows = []
    for row in ds:
        s = row["translation"][args.src]
        t = row["translation"][args.tgt]
        if s in seen:
            continue
        if len(s.split()) > args.max_words or len(t.split()) > args.max_words:
            continue
        seen.add(s)
        out_rows.append({"src_text": s, "tgt_text": t,
                         "src_lang": args.src, "tgt_lang": args.tgt})
    random.Random(args.seed).shuffle(out_rows)
    if args.limit:
        out_rows = out_rows[:args.limit]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out_rows)} pairs -> {args.out}")


if __name__ == "__main__":
    main()
