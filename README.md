# multilingual-nmt-production

Production multilingual neural machine translation. Fine-tunes mBART-50 on small custom corpora (Tatoeba pairs) and serves batch translation via FastAPI.

## Problem

Off-the-shelf NMT systems often miss domain phrasing and rare language pairs.
We fine-tune `facebook/mbart-large-50-many-to-many-mmt` on a curated corpus
and ship it behind a typed batch endpoint with language detection.

## Stack

* PyTorch 2.4 + transformers 4.44
* HF datasets, accelerate
* FastAPI for serving
* MLflow for experiment tracking
* sacrebleu (spBLEU + chrF) for eval
* Docker + ECS Fargate for deploy

## Quickstart (rough)

```
pip install -r requirements.txt
python -m src.train --config configs/finetune.yaml
python -m src.api.main
```

More to come.
