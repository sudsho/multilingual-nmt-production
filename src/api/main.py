"""FastAPI app for batch translation."""

from __future__ import annotations

import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/translate", response_model=TranslateResponse)
def translate_endpoint(req: TranslateRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="empty input")
    raise HTTPException(status_code=501, detail="not yet wired")
