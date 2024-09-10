"""FastAPI app for batch translation."""

from __future__ import annotations

import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.api.lid import detect_lang
from src.preprocess import to_mbart_code


app = FastAPI(title="multilingual-nmt-production")


class TranslateRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1)
    src_lang: Optional[str] = None
    tgt_lang: str
    num_beams: int = 5
    length_penalty: float = 1.0


class TranslateResponse(BaseModel):
    translations: List[str]
    src_lang: List[str]


_MODEL = None
_TOK = None


def _ensure_model():
    global _MODEL, _TOK
    if _MODEL is None:
        from src.model import load_model_and_tokenizer
        ckpt = os.environ.get("MNMT_CHECKPOINT", "facebook/mbart-large-50-many-to-many-mmt")
        _MODEL, _TOK = load_model_and_tokenizer(ckpt)
    return _MODEL, _TOK


def _resolve_src(text: str, override: Optional[str]) -> str:
    if override:
        return to_mbart_code(override)
    short = detect_lang(text) or "en"
    return to_mbart_code(short)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/translate", response_model=TranslateResponse)
def translate_endpoint(req: TranslateRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="empty input")
    from src.translate import translate
    model, tok = _ensure_model()
    src_codes = [_resolve_src(t, req.src_lang) for t in req.texts]
    tgt_code = to_mbart_code(req.tgt_lang)

    # If all input texts share a src lang, batch them. Else fall back to per-text loop.
    out: List[str] = []
    if len(set(src_codes)) == 1:
        out = translate(model, tok, req.texts, src_codes[0], tgt_code,
                        num_beams=req.num_beams, length_penalty=req.length_penalty)
    else:
        for text, src_code in zip(req.texts, src_codes):
            out.extend(translate(model, tok, [text], src_code, tgt_code,
                                 num_beams=req.num_beams, length_penalty=req.length_penalty))
    return TranslateResponse(translations=out, src_lang=src_codes)
