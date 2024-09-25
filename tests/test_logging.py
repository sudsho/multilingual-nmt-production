import json
import logging

from src.logging_setup import JsonFormatter, setup


def test_formatter_emits_json():
    rec = logging.LogRecord(
        name="t",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="hello",
        args=(),
        exc_info=None,
    )
    out = JsonFormatter().format(rec)
    payload = json.loads(out)
    assert payload["level"] == "INFO"
    assert payload["msg"] == "hello"


def test_setup_attaches_one_handler():
    root = setup()
    assert len(root.handlers) == 1
