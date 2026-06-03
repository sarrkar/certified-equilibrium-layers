# CASCADE Makefile. Headline numbers reproduce via 'make repro'.
.PHONY: help install dev lint fmt test verify repro clean

help:
	@echo "Targets:"
	@echo "  install   pip install -e . (runtime deps)"
	@echo "  dev       pip install -e .[dev] + pre-commit install"
	@echo "  lint      ruff check + ruff format --check + em-dash scan"
	@echo "  fmt       ruff format + ruff check --fix"
	@echo "  test      pytest"
	@echo "  verify    run all numerical verification scripts"
	@echo "  repro     reproduce all headline numbers from cached data"
	@echo "  clean     remove caches and build artifacts"

install:
	python -m pip install -e .

dev:
	python -m pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check .
	ruff format --check .
	git ls-files -z | xargs -0 python scripts/check_no_em_dash.py

fmt:
	ruff format .
	ruff check . --fix

test:
	pytest -q

# Each script under verify/ corresponds to a theorem and exits non-zero on
# bound violation. Add new entries here as theorems are added.
verify:
	python verify/containment.py --seed 0

repro: verify test
	@echo "Headline numbers reproduced."

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
