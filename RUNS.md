# Run notes

Recording fine-tune attempts so we don't repeat the same mistakes.

## en-hi v0 (2024-09-18)

* Data: tatoeba en-hi, 38k pairs after dedupe
* Steps: 5 epochs, bs=4, ga=4, lr=3e-5, bf16
* spBLEU: 18.2
* chrF: 41.4
* Notes: hindi side benefits from a slightly higher length penalty (1.1)
  during decode. Default of 1.0 was clipping a few outputs short.

## en-hi v1 (planned)

* Try bigger context (max_source_length=128) and longer beams (num_beams=6).
* Add backtranslation of unpaired english news to bump diversity.
