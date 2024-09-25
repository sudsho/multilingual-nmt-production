import os
from pathlib import Path
from unittest import mock

from src.config_loader import _coerce, load


def test_coerce_types():
    assert _coerce("true") is True
    assert _coerce("false") is False
    assert _coerce("42") == 42
    assert _coerce("3.14") == 3.14
    assert _coerce("hello") == "hello"


def test_load_yaml(tmp_path: Path):
    p = tmp_path / "c.yaml"
    p.write_text("train:\n  lr: 0.001\n  epochs: 3\n", encoding="utf-8")
    cfg = load(str(p))
    assert cfg["train"]["lr"] == 0.001
    assert cfg["train"]["epochs"] == 3


def test_env_overrides_take_priority(tmp_path: Path):
    p = tmp_path / "c.yaml"
    p.write_text("train:\n  lr: 0.001\n", encoding="utf-8")
    with mock.patch.dict(os.environ, {"MNMT_TRAIN__LR": "0.0005"}, clear=False):
        cfg = load(str(p))
    assert cfg["train"]["lr"] == 0.0005
