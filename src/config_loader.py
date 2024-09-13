"""Tiny config loader: yaml file + optional env-var overrides like `MNMT_TRAIN_LR=1e-5`."""

from __future__ import annotations

import os
from typing import Any, Dict

import yaml


def load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return _apply_env(cfg)


def _apply_env(cfg: Dict[str, Any], prefix: str = "MNMT") -> Dict[str, Any]:
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix + "_"):
            continue
        path = env_key[len(prefix) + 1:].lower().split("__")
        node = cfg
        for p in path[:-1]:
            node = node.setdefault(p, {})
        node[path[-1]] = _coerce(env_val)
    return cfg


def _coerce(val: str) -> Any:
    if val.lower() in {"true", "false"}:
        return val.lower() == "true"
    try:
        if "." in val or "e" in val.lower():
            return float(val)
        return int(val)
    except ValueError:
        return val
