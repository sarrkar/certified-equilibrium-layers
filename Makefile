# CASCADE Makefile. Headline numbers reproduce via 'make repro'.
.PHONY: help install dev lint fmt test train verify assert-no-synthetic repro repro-aaai clean

help:
	@echo "Targets:"
	@echo "  install            pip install -e . (runtime deps)"
	@echo "  dev                pip install -e .[dev] + pre-commit install"
	@echo "  lint               ruff check + ruff format --check + em-dash scan"
	@echo "  fmt                ruff format + ruff check --fix"
	@echo "  test               pytest"
	@echo "  train              smoke-train CASCADE on the fixture graph (hydra)"
	@echo "  verify             run all numerical verification scripts"
	@echo "  assert-no-synthetic  fail if generated data sits on an AAAI reported-number path"
	@echo "  repro              reproduce all headline numbers from cached data"
	@echo "  repro-aaai         assert-no-synthetic, then aggregate the AAAI paper numbers"
	@echo "  clean              remove caches and build artifacts"

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

train:
	python train.py

# Each script under verify/ corresponds to a theorem and exits non-zero on
# bound violation. Add new entries here as theorems are added.
verify:
	python verify/containment.py --seed 0

# Hard data policy (CLAUDE.md): no generated data on any AAAI reported-number path.
assert-no-synthetic:
	python scripts/assert_no_synthetic.py

repro: verify test
	@echo "Headline numbers reproduced."

# AAAI paper numbers. The no-synthetic assertion is a hard precondition: the
# aggregation does not run unless the data policy holds.
repro-aaai: assert-no-synthetic
	python experiments/aaai/export_paper_numbers.py --seed 0
	@echo "AAAI paper numbers aggregated (real-data only)."

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
