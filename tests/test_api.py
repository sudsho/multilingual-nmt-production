"""API integration tests with model + lid mocked."""

from unittest import mock

from fastapi.testclient import TestClient

import src.api.main as api


def _patched_translate(model, tok, texts, src, tgt, **kw):
    return [f"[{tgt}] {t}" for t in texts]


def test_health():
    client = TestClient(api.app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_metrics_initial_zero():
    client = TestClient(api.app)
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert "translate_calls" in body


def test_translate_happy_path():
    client = TestClient(api.app)
    with mock.patch.object(api, "_ensure_model", lambda: ("M", "T")), \
         mock.patch("src.translate.translate", _patched_translate), \
         mock.patch("src.api.main.detect_lang", lambda t: "en"):
        r = client.post("/translate", json={"texts": ["hi", "world"], "tgt_lang": "hi"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["translations"]) == 2
    assert all(t.startswith("[hi_IN]") for t in body["translations"])


def test_translate_empty_text_field_rejected():
    client = TestClient(api.app)
    r = client.post("/translate", json={"texts": [], "tgt_lang": "hi"})
    # pydantic min_length=1 -> 422
    assert r.status_code == 422
