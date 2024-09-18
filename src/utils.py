"""Small helpers used across the package."""

from __future__ import annotations

import random
from typing import Iterable, List, TypeVar

T = TypeVar("T")


def chunks(items: Iterable[T], size: int) -> Iterable[List[T]]:
    buf: List[T] = []
    for it in items:
        buf.append(it)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
