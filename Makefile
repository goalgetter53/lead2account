PY ?= python
export PYTHONPATH := src

.PHONY: install lint type test check fetch select gleif build experiment sample label hand-eval report all

install:
	$(PY) -m pip install -e '.[dev]'

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

type:
	$(PY) -m mypy

test:
	$(PY) -m pytest -q

check: lint type test

# ---- data (network; results are committed so this is optional)
fetch:
	$(PY) -m l2a.sources.wikidata --out data/raw/wikidata_lei_companies.jsonl.gz
	$(PY) -c "from pathlib import Path; from l2a.sources.gleif import download_rr; download_rr(Path('data/raw/gleif_rr.csv.zip'))"

select:
	$(PY) -m l2a.sources.wikidata --only core,website --out data/raw/wikidata_seed_partial.jsonl.gz
	$(PY) -m l2a.dataset select --wikidata data/raw/wikidata_seed_partial.jsonl.gz

gleif:
	$(PY) -m l2a.sources.gleif --leis data/raw/pool_leis.txt --out data/raw/gleif_level1.jsonl.gz

build:
	$(PY) -m l2a.dataset build

# ---- modelling (offline)
experiment:
	$(PY) -m l2a.experiment

sample:
	$(PY) -m l2a.sampling

label:
	$(PY) -m l2a.labeller.app

hand-eval:
	$(PY) -m l2a.hand_eval

report:
	$(PY) -m l2a.report

all: build experiment hand-eval report
