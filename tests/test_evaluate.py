from src.evaluate import chrf, spbleu


def test_perfect_match_high_scores():
    hyps = ["hello world", "the cat sat"]
    refs = ["hello world", "the cat sat"]
    bleu = spbleu(hyps, refs)
    cf = chrf(hyps, refs)
    assert bleu > 80
    assert cf > 90


def test_terrible_match_low_scores():
    hyps = ["abcdef", "ghijkl"]
    refs = ["mnop qrs", "tuv wxyz"]
    bleu = spbleu(hyps, refs)
    cf = chrf(hyps, refs)
    assert bleu < 10
    assert cf < 30
