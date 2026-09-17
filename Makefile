# KAIROS: local reproduction targets (design rule 0.5). Windows users without make: python scripts/build_all.py
PY ?= .venv/Scripts/python
ifeq ($(wildcard .venv/bin/python),.venv/bin/python)
PY := .venv/bin/python
endif

.PHONY: all test lint scan schemas scenarios train evaluate quick deploy

all: test scan schemas scenarios train evaluate
quick: ; $(PY) scripts/build_all.py --quick
test: ; $(PY) -m pytest -q
lint: ; $(PY) -m ruff check .
scan: ; $(PY) scripts/privacy_scan.py --allow-private-dirs
schemas: ; $(PY) scripts/export_schemas.py
scenarios: ; $(PY) services/jobs/cli.py scenarios --all
train: ; $(PY) services/jobs/cli.py train
evaluate: ; $(PY) services/jobs/cli.py evaluate --all
deploy: ; bash scripts/deploy.sh
