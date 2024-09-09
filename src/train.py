"""Fine-tuning loop using HF accelerate and MLflow.

Usage:
    accelerate launch -m src.train --config configs/finetune.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/finetune.yaml")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    print("loaded config:", cfg["model"]["name"])
    # full loop comes later
    raise NotImplementedError


if __name__ == "__main__":
    main()
