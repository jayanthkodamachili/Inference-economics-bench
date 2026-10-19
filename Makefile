.PHONY: install lint test test-fast bench figures receipts verify clean

# Every target runs through one interpreter. A bare pytest picks whichever console script
# is first on PATH, which can belong to a different Python than the one holding the
# editable install, and the resulting error looks like a missing package.
PYTHON ?= python3

install:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest --cov=inference_economics --cov-report=term-missing --cov-fail-under=100

# Skips the tests that train the transformer.
test-fast:
	$(PYTHON) -m pytest -m "not slow"

# Rerun the measurements and rewrite docs/experiments. Not part of verify: latency is not
# reproducible, and a check that demands it match fails at random. See ADR-003.
bench:
	$(PYTHON) experiments/e1_the_frontier.py
	$(PYTHON) experiments/e2_does_batching_change_the_answer.py
	$(PYTHON) experiments/e3_the_budget_decides.py
	$(PYTHON) experiments/e4_when_does_it_pay.py

figures:
	$(PYTHON) tools/render_diagram.py
	$(PYTHON) tools/capture_cli.py

receipts:
	$(PYTHON) tools/check_reproducible.py
	$(PYTHON) tools/collect_metrics.py
	$(PYTHON) tools/check_numbers.py

verify: lint test receipts

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov coverage.json junit.xml .coverage artifacts
