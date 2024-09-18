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


def collate(batch, pad_id: int):
    from src.collator import PadCollator
    return PadCollator(pad_id)(batch)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    import mlflow
    import torch
    from accelerate import Accelerator
    from torch.utils.data import DataLoader

    from src.model import load_model_and_tokenizer

    from src.utils import set_seed
    set_seed(int(cfg["train"].get("seed", 7)))

    accelerator = Accelerator()
    model, tok = load_model_and_tokenizer(cfg["model"]["name"])

    ds = build_dataset(cfg, tok)

    pad_id = tok.pad_token_id
    loader = DataLoader(
        ds,
        batch_size=cfg["train"]["per_device_train_batch_size"],
        shuffle=True,
        collate_fn=lambda b: collate(b, pad_id),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["learning_rate"]),
                                  weight_decay=cfg["train"]["weight_decay"])
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    out = Path(cfg["train"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    n_epochs = cfg["train"]["num_train_epochs"]
    log_every = cfg["train"]["logging_steps"]

    if accelerator.is_main_process:
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
        mlflow.start_run(run_name=cfg["mlflow"].get("run_name"))
        flat = {f"train.{k}": v for k, v in cfg["train"].items()}
        flat.update({f"data.{k}": v for k, v in cfg["data"].items()})
        flat.update({f"model.{k}": v for k, v in cfg["model"].items()})
        mlflow.log_params({k: str(v) for k, v in flat.items()})

    step = 0
    for epoch in range(n_epochs):
        model.train()
        for batch in loader:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            if step % log_every == 0 and accelerator.is_main_process:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
                mlflow.log_metric("loss", loss.item(), step=step)
            step += 1

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(out)
        tok.save_pretrained(out)
        mlflow.end_run()


if __name__ == "__main__":
    main()
