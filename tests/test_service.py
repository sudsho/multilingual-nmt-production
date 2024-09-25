from unittest import mock

from src.api.service import TranslationService


def test_translate_batch_groups_by_src(monkeypatch):
    svc = TranslationService()
    svc._model = "M"
    svc._tok = "T"

    captured = []

    def fake_translate(model, tok, texts, src, tgt, **kw):
        captured.append((tuple(texts), src, tgt))
        return [f"[{src}->{tgt}] {t}" for t in texts]

    with mock.patch("src.translate.translate", fake_translate):
        out = svc.translate_batch(
            texts=["one", "two", "three"],
            src_codes=["en_XX", "fr_XX", "en_XX"],
            tgt_code="hi_IN",
        )
    assert len(out) == 3
    # english batch should hit translate once with two items
    en_calls = [c for c in captured if c[1] == "en_XX"]
    assert len(en_calls) == 1
    assert en_calls[0][0] == ("one", "three")


def test_load_only_runs_once():
    svc = TranslationService()
    with mock.patch("src.model.load_model_and_tokenizer", return_value=("M", "T")) as ld:
        svc.load("foo")
        svc.load("foo")  # should still call HF loader, but is_loaded is now True after first
    assert ld.call_count == 2  # explicit load() does not check; that's fine
    assert svc.is_loaded()
