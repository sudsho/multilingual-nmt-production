"""End-to-end evaluation runner.

Loads a fine-tuned checkpoint, decodes a parallel jsonl, computes spBLEU+chrF.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--src", required=True)
    p.add_argument("--tgt", required=True)
    p.add_argument("--input", required=True, help="parallel jsonl with src_text/tgt_text")
    p.add_argument("--num-beams", type=int, default=5)
    p.add_argument("--length-penalty", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    from src.evaluate import chrf, spbleu
    from src.model import load_model_and_tokenizer
    from src.preprocess import to_mbart_code
    from src.translate import translate

    model, tok = load_model_and_tokenizer(args.ckpt)
    model.eval()

    src_code = to_mbart_code(args.src)
    tgt_code = to_mbart_code(args.tgt)

    srcs: List[str] = []
    refs: List[str] = []
    with Path(args.input).open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            srcs.append(row["src_text"])
            refs.append(row["tgt_text"])

    from src.utils import chunks
    hyps: List[str] = []
    for chunk in chunks(srcs, args.batch_size):
        hyps.extend(translate(model, tok, chunk, src_code, tgt_code,
                              num_beams=args.num_beams, length_penalty=args.length_penalty))

    bleu = spbleu(hyps, refs)
    cf = chrf(hyps, refs)
    print(json.dumps({"spbleu": bleu, "chrf": cf, "n": len(hyps)}, indent=2))


if __name__ == "__main__":
    main()
