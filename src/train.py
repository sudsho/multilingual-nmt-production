"""Fine-tuning loop using HF accelerate and MLflow.

Usage:
    accelerate launch -m src.train --config configs/finetune.yaml
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict

import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/finetune.yaml")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataset(cfg: dict, tokenizer):
    from datasets import Dataset
    from src.data import load_pairs
    from src.preprocess import encode_pair

    pairs = list(load_pairs(
        src=cfg["data"]["src_lang"],
        tgt=cfg["data"]["tgt_lang"],
        jsonl_path=cfg["data"].get("jsonl_path"),
    ))
    rows: Dict[str, list] = {"input_ids": [], "attention_mask": [], "labels": []}
    for p in pairs:
        enc = encode_pair(
            tokenizer,
            p.src_text,
            p.tgt_text,
            cfg["data"]["src_lang"],
            cfg["data"]["tgt_lang"],
            max_source_length=cfg["model"]["max_source_length"],
            max_target_length=cfg["model"]["max_target_length"],
        )
        rows["input_ids"].append(enc["input_ids"])
        rows["attention_mask"].append(enc["attention_mask"])
        rows["labels"].append(enc["labels"])
    return Dataset.from_dict(rows)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    from src.model import load_model_and_tokenizer
    model, tok = load_model_and_tokenizer(cfg["model"]["name"])

    ds = build_dataset(cfg, tok)
    print(f"loaded {len(ds)} training rows")

    out = Path(cfg["train"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {out}")
    # accelerate loop wired in next commit
    raise NotImplementedError("training loop not finished yet")


if __name__ == "__main__":
    main()
