import json
from pathlib import Path

from src.data import Pair, load_pairs


def test_jsonl_loader(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    rows = [
        {"src_text": "hello", "tgt_text": "namaste",
         "src_lang": "en", "tgt_lang": "hi"},
        {"src_text": "good night", "tgt_text": "shubh ratri",
         "src_lang": "en", "tgt_lang": "hi"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    pairs = list(load_pairs(src="en", tgt="hi", jsonl_path=p))
    assert len(pairs) == 2
    assert all(isinstance(x, Pair) for x in pairs)
    assert pairs[0].src_text == "hello"
    assert pairs[1].tgt_text == "shubh ratri"


def test_blank_lines_skipped(tmp_path: Path):
    p = tmp_path / "y.jsonl"
    p.write_text(
        '{"src_text": "a", "tgt_text": "b", "src_lang": "en", "tgt_lang": "hi"}\n\n'
        '{"src_text": "c", "tgt_text": "d", "src_lang": "en", "tgt_lang": "hi"}\n',
        encoding="utf-8",
    )
    assert len(list(load_pairs(src="en", tgt="hi", jsonl_path=p))) == 2
