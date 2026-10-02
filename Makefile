.PHONY: install lint typecheck test cov reproduce clean

install:
	pip install -e ".[dev]"

lint:
	ruff check .

typecheck:
	mypy

test:
	pytest -q

cov:
	pytest -q --cov --cov-report=term-missing

reproduce:
	bash scripts/reproduce.sh

clean:
	rm -rf outputs .pytest_cache .ruff_cache .mypy_cache .coverage
