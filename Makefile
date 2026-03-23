PYTHON ?= python3

.PHONY: smoke test

smoke:
	PYTHONPATH=src $(PYTHON) scripts/smoke_test.py

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests
