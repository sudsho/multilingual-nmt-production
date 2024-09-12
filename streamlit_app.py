"""Streamlit demo: paste text, pick target language, see translation."""

from __future__ import annotations

import os

import streamlit as st


TGT_OPTIONS = ["en", "hi", "fr", "de", "es", "it", "ja", "ko", "zh", "ar", "ru", "pt"]


def _load():
    from src.api.lid import detect_lang
    from src.model import load_model_and_tokenizer
    from src.preprocess import to_mbart_code
    from src.translate import translate

    ckpt = os.environ.get("MNMT_CHECKPOINT", "facebook/mbart-large-50-many-to-many-mmt")
    model, tok = load_model_and_tokenizer(ckpt)
    return model, tok, detect_lang, to_mbart_code, translate


@st.cache_resource
def cached():
    return _load()


def main():
    st.set_page_config(page_title="multilingual nmt", layout="centered")
    st.title("multilingual nmt - mBART-50")

    text = st.text_area("input text", height=180, value="Hello, world.")
    tgt = st.selectbox("target language", TGT_OPTIONS, index=1)
    beams = st.slider("beam size", 1, 8, 5)
    lp = st.slider("length penalty", 0.5, 1.5, 1.0, 0.05)

    if st.button("translate"):
        model, tok, detect, to_code, translate = cached()
        if not text.strip():
            st.warning("enter some text")
            return
        src_short = detect(text) or "en"
        src_code = to_code(src_short)
        tgt_code = to_code(tgt)
        with st.spinner("translating..."):
            out = translate(model, tok, [text], src_code, tgt_code,
                            num_beams=beams, length_penalty=lp)
        st.markdown(f"**detected source:** `{src_short}`")
        st.success(out[0])


if __name__ == "__main__":
    main()
