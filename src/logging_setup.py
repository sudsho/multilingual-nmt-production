"""Structured json logging setup for the api and CLI scripts."""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any, Dict


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for k, v in record.__dict__.items():
            if k.startswith("_") or k in {"args", "msg", "levelname", "levelno",
                                          "pathname", "filename", "module", "exc_info",
                                          "exc_text", "stack_info", "lineno", "funcName",
                                          "created", "msecs", "relativeCreated", "thread",
                                          "threadName", "processName", "process", "name",
                                          "message"}:
                continue
            payload[k] = v
        return json.dumps(payload, ensure_ascii=False)


def setup(level: int = logging.INFO):
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(h)
    root.setLevel(level)
    return root
