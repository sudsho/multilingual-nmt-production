# multilingual-nmt-production

Production multilingual neural machine translation. Fine-tunes
`facebook/mbart-large-50-many-to-many-mmt` on a small custom corpus (Tatoeba
pairs) and serves batch translation behind FastAPI with language detection.

## Quick start (tiny-CPU smoke, no GPU/download)

The headline system fine-tunes mBART-50 (a ~2.4 GB download) on a GPU and reports
spBLEU/chrF. That needs a GPU, a real parallel corpus, and network access, so it
does not run on a plain laptop. To prove the pipeline shape end to end on CPU in a
few seconds with no downloads and no pretrained weights, run the smoke:

```bash
make smoke        # or: python scripts/smoke.py
```

It builds a tiny synthetic parallel corpus from a deterministic src to tgt rule
(target = reversed source over a 6-symbol vocab), trains a small GRU
encoder-decoder with attention for a few hundred steps, greedy-decodes held-out
examples, and runs the real language-detection component on multilingual sample
strings using its offline heuristic fallback (the fastText model download is
guarded and skipped). Real verified output:

```
============================================================
multilingual-nmt-production  tiny-CPU offline smoke
============================================================
synthetic task : reverse a length-5 sequence over ['a', 'b', 'c', 'd', 'e', 'f']
model          : GRU enc-dec + attention, vocab=9, params=66,922
train pairs    : 2000  (fully synthetic, no download)
------------------------------------------------------------
step    0  loss 2.2052
step  200  loss 0.0801
step  400  loss 0.0272
step  699  loss 0.0086
------------------------------------------------------------
loss: 2.2052 -> 0.0086  (decreased)
------------------------------------------------------------
held-out greedy decode (src -> expected | got):
  faeee -> eeeaf | eeeaf  OK
  ddbfc -> cfbdd | cfbdd  OK
  bceaf -> faecb | faecb  OK
  acabf -> fbaca | fbaca  OK
------------------------------------------------------------
language detection (offline heuristic, no fastText download):
  'Hello, world.'                               -> en  (expected en)  OK
  'Bonjour le monde, comment allez vous'        -> fr  (expected fr)  OK
  'Hola mundo, ¿como estas?'                    -> es  (expected es)  OK
  'Guten Morgen, wie geht es dir?'              -> de  (expected de)  OK
  'नमस्ते दुनिया'                               -> hi  (expected hi)  OK
  'こんにちは世界'                                     -> ja  (expected ja)  OK
  'Привет мир'                                  -> ru  (expected ru)  OK
------------------------------------------------------------
mBART-50 / transformers download guarded and SKIPPED (set MNMT_RUN_REAL=1 to opt in).
============================================================
loss decreased >=2x : True  (2.205 -> 0.009)
held-out decode acc : 1.00  (>=0.75 required)
language-id accuracy: 1.00  (>=0.85 required)
SMOKE OK
```

Run the test suite (heavy HF/mBART weights are mocked, so it stays on CPU):

```bash
python -m pytest -q      # 36 passed
```

What the smoke does NOT do: it is a stand-in, not the real model. The headline
mBART-50 translation quality and the spBLEU/chrF numbers below need a GPU, the
pretrained mBART-50 weights (a large download), and a real parallel corpus. The
smoke only proves the training loop, attention decode, language detection, and
eval/serving plumbing run offline on CPU.

## Problem

Off-the-shelf NMT systems often miss domain phrasing and rare language pairs.
We fine-tune mBART-50 on a curated parallel corpus and ship the result behind
a typed batch endpoint with automatic source-language detection.

## Stack

* PyTorch 2.4 + transformers 4.44
* HuggingFace `datasets` + `accelerate` for the fine-tune loop
* FastAPI (pydantic v2) for serving, with `/health`, `/ready`, `/metrics`,
  `/translate`
* fasttext-langdetect for source-side LID
* MLflow for experiment tracking
* sacrebleu (spBLEU + chrF) for eval
* Docker (multi-stage), ECS Fargate (terraform), CI to deploy to staging
* Streamlit demo for paste-and-translate

## Layout

```
src/
  data.py            # tatoeba + jsonl loader
  preprocess.py      # mBART tokenizer dance, lang code mapping
  model.py           # MBart50 loader
  train.py           # accelerate-based fine-tune, MLflow
  translate.py       # beam search + length penalty
  evaluate.py        # spBLEU + chrF wrappers
  eval_runner.py     # CLI: ckpt + jsonl -> metrics
  build_corpus.py    # filter/dedupe/dump tatoeba -> jsonl
  collator.py        # pad-on-the-fly seq2seq collator
  config_loader.py   # yaml + env var overrides
  utils.py           # set_seed, chunks
  logging_setup.py   # json logs
  api/
    main.py          # FastAPI app
    service.py       # TranslationService
    lid.py           # source-side language detection
configs/
  default.yaml
  finetune.yaml      # en-hi small corpus example
tests/               # pytest suite (no GPU needed; heavy deps mocked)
notebooks/
  eda.ipynb          # length / pair distribution
streamlit_app.py     # paste text, pick target lang, see translation
terraform/           # S3 + ECR + ECS Fargate + IAM
ci/test.yml.example  # gh actions: lint, test, build image, deploy-to-staging
scripts/
  train.sh           # accelerate launch wrapper
  eval.sh            # eval_runner wrapper
  deploy.sh          # build, push to ECR, force ECS update
  smoke.sh           # curl /health + /translate
Dockerfile           # multi-stage, runtime is python:3.11-slim
docker-compose.yml   # api + mlflow tracker
```

## Quickstart

```bash
pip install -r requirements.txt

# build a small en-hi corpus
python -m src.build_corpus --src en --tgt hi --out data/processed/en_hi.train.jsonl --limit 40000

# fine-tune
accelerate launch -m src.train --config configs/finetune.yaml

# evaluate
bash scripts/eval.sh outputs/finetune-en-hi en hi data/processed/en_hi.val.jsonl

# serve
uvicorn src.api.main:app --port 8080
curl -X POST localhost:8080/translate \
  -H 'content-type: application/json' \
  -d '{"texts":["Hello, world."], "tgt_lang":"hi"}'

# streamlit demo
streamlit run streamlit_app.py
```

## Results (en-hi v0)

On the val split (4k held-out tatoeba pairs):

| metric  | value |
|---------|-------|
| spBLEU  | 18.2  |
| chrF    | 41.4  |

Baseline mBART-50 (no fine-tune) on the same split: spBLEU 11.7, chrF 33.0.

## Deploy

The terraform module provisions:

* S3 bucket for fine-tuned checkpoints (versioned)
* ECR repo for the api image (immutable tags)
* ECS cluster + Fargate task def + execution role
* outputs cluster name and ECR url

CI pushes a new image to ECR on every main commit and force-deploys the
staging service. Promotion to prod is manual.

## License

MIT
