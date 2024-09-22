"""LID tests with the heavy model patched out."""

from unittest import mock

from src.api import lid


def test_empty_string_returns_none():
    assert lid.detect_lang("") is None
    assert lid.detect_lang("   ") is None


def test_normal_path_returns_lang():
    fake = mock.MagicMock(return_value={"lang": "en", "score": 0.99})
    with mock.patch.object(lid, "_detector", lambda: fake):
        out = lid.detect_lang("Hello, world.")
        assert out == "en"
        fake.assert_called_once()


def test_detector_exception_returns_none():
    def boom(text, low_memory):
        raise RuntimeError("boom")
    with mock.patch.object(lid, "_detector", lambda: boom):
        assert lid.detect_lang("hello") is None
