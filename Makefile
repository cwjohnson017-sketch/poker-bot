PY ?= python

.PHONY: test test-all lint format engine match

test:  ## CPU test suite (skips tests marked slow)
	$(PY) -m pytest -q -m "not slow"

test-all:  ## everything, including slow tests
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check pokerbot tests scripts

format:
	$(PY) -m ruff format pokerbot tests scripts
	$(PY) -m ruff check --fix .

engine:  ## build the Rust engine into the active venv
	cd engine && maturin develop --release

match:
	$(PY) scripts/play_match.py --a always_call --b equity --hands 2000 --duplicate --seed 0
