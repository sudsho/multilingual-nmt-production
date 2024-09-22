import pytest

from src.preprocess import LANGS, to_mbart_code


def test_short_to_long():
    assert to_mbart_code("en") == "en_XX"
    assert to_mbart_code("hi") == "hi_IN"
    assert to_mbart_code("zh") == "zh_CN"


def test_passthrough_long_form():
    for code in LANGS.values():
        assert to_mbart_code(code) == code


def test_unknown_language_raises():
    with pytest.raises(ValueError):
        to_mbart_code("xx")
