"""Translate tests with the heavy model + tokenizer faked with tiny tensors."""

from src.translate import translate


class FakeTokenizer:
    def __init__(self):
        self.src_lang = None
        self.lang_code_to_id = {"en_XX": 1, "hi_IN": 2}

    def __call__(self, texts, return_tensors=None, padding=None, truncation=None):
        import torch
        return {
            "input_ids": torch.tensor([[3] * len(texts)]),
            "attention_mask": torch.tensor([[1] * len(texts)]),
        }

    def batch_decode(self, ids, skip_special_tokens=True):
        return [f"out-{i}" for i in range(len(ids))]


class FakeModel:
    device = "cpu"

    def generate(self, **kw):
        return [[1, 2, 3]] * len(kw["input_ids"])


def test_translate_calls_generate():
    tok = FakeTokenizer()
    model = FakeModel()
    out = translate(model, tok, ["hi there"], "en_XX", "hi_IN", num_beams=4)
    assert tok.src_lang == "en_XX"
    assert isinstance(out, list)
    assert all(isinstance(s, str) for s in out)


def test_translate_uses_target_lang_code():
    tok = FakeTokenizer()
    model = FakeModel()
    captured = {}

    def fake_generate(**kw):
        captured.update(kw)
        return [[1, 2, 3]]

    model.generate = fake_generate
    translate(model, tok, ["hello"], "en_XX", "hi_IN", num_beams=5, length_penalty=1.1)
    assert captured["forced_bos_token_id"] == 2
    assert captured["num_beams"] == 5
    assert captured["length_penalty"] == 1.1
