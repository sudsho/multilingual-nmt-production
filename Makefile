.PHONY: install fmt lint test smoke serve train eval docker

install:
	pip install -r requirements.txt

smoke:
	python scripts/smoke.py

fmt:
	python -m ruff format src tests

lint:
	python -m ruff check src tests

test:
	python -m pytest

serve:
	uvicorn src.api.main:app --reload --port 8080

train:
	bash scripts/train.sh configs/finetune.yaml

eval:
	bash scripts/eval.sh

docker:
	docker build -t mnmt-api:dev .
