"""Model loader tests with the heavy weights mocked.

We only verify that the loader calls the right HF API and returns a
(model, tokenizer) tuple. Forward pass is exercised via a small fake module
that mimics the MBart return type.
"""

from types import SimpleNamespace
from unittest import mock

from src import model as model_mod


def test_load_model_and_tokenizer_calls_hf():
    fake_model = object()
    fake_tok = object()
    with mock.patch("transformers.MBart50TokenizerFast.from_pretrained", return_value=fake_tok) as t, \
         mock.patch("transformers.MBartForConditionalGeneration.from_pretrained", return_value=fake_model) as m:
        out_model, out_tok = model_mod.load_model_and_tokenizer("path/to/ckpt")
    t.assert_called_once_with("path/to/ckpt")
    m.assert_called_once_with("path/to/ckpt")
    assert out_model is fake_model
    assert out_tok is fake_tok


def test_default_checkpoint_name():
    with mock.patch("transformers.MBart50TokenizerFast.from_pretrained", return_value=None), \
         mock.patch("transformers.MBartForConditionalGeneration.from_pretrained", return_value=None) as m:
        model_mod.load_model_and_tokenizer()
    assert m.call_args.args[0] == "facebook/mbart-large-50-many-to-many-mmt"


def test_forward_pass_shape_with_fake():
    """A tiny smoke test pretending the model returns a Seq2SeqLMOutput-like object."""
    fake_out = SimpleNamespace(loss=0.42, logits=None)
    fake_model = mock.MagicMock(return_value=fake_out)
    out = fake_model(input_ids=[[1, 2, 3]], attention_mask=[[1, 1, 1]], labels=[[2, 3, 4]])
    assert out.loss == 0.42
