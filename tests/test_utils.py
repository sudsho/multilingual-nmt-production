from src.utils import chunks, set_seed


def test_chunks_basic():
    out = list(chunks([1, 2, 3, 4, 5], 2))
    assert out == [[1, 2], [3, 4], [5]]


def test_chunks_empty():
    assert list(chunks([], 3)) == []


def test_chunks_exact_multiple():
    assert list(chunks([1, 2, 3, 4], 2)) == [[1, 2], [3, 4]]


def test_set_seed_runs_without_optional_deps():
    set_seed(42)
