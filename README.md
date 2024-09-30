# multilingual-nmt-production

Production multilingual neural machine translation. Fine-tunes
`facebook/mbart-large-50-many-to-many-mmt` on a small custom corpus (Tatoeba
pairs) and serves batch translation behind FastAPI with language detection.

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
