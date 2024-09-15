"""Translation entry point. Beam search with length penalty."""

from __future__ import annotations

from typing import List


def translate(model, tokenizer, texts: List[str], src_lang: str, tgt_lang: str,
              num_beams: int = 5, length_penalty: float = 1.0,
              max_new_tokens: int = 128, no_repeat_ngram_size: int = 3,
              early_stopping: bool = True) -> List[str]:
    """Translate a batch of texts. Caller passes mBART codes (e.g. en_XX)."""
    import torch

    tokenizer.src_lang = src_lang
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    if hasattr(model, "device"):
        enc = {k: v.to(model.device) for k, v in enc.items()}
    forced_bos = tokenizer.lang_code_to_id[tgt_lang]
    with torch.inference_mode():
        out = model.generate(
            **enc,
            forced_bos_token_id=forced_bos,
            num_beams=num_beams,
            length_penalty=length_penalty,
            max_new_tokens=max_new_tokens,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
        )
    return tokenizer.batch_decode(out, skip_special_tokens=True)
