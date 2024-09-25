"""Shared fixtures.

Most tests skip the heavy HF model + tokenizer by patching, so the test suite
runs in a few seconds. We pin a deterministic seed here for any randomness.
"""

import random

import pytest


@pytest.fixture(autouse=True)
def _seed():
    random.seed(7)
    yield
