"""Model loader for mBART-50 many-to-many."""

from __future__ import annotations

from typing import Optional, Tuple

from transformers import MBart50TokenizerFast, MBartForConditionalGeneration


def load_model_and_tokenizer(
    name: str = "facebook/mbart-large-50-many-to-many-mmt",
    dtype: Optional[str] = None,
) -> Tuple[MBartForConditionalGeneration, MBart50TokenizerFast]:
    tok = MBart50TokenizerFast.from_pretrained(name)
    kwargs = {}
    if dtype == "bf16":
        import torch
        kwargs["torch_dtype"] = torch.bfloat16
    elif dtype == "fp16":
        import torch
        kwargs["torch_dtype"] = torch.float16
    model = MBartForConditionalGeneration.from_pretrained(name, **kwargs)
    return model, tok
