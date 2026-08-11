# Development tasks for mesoscopy.

.DEFAULT_GOAL := help
.PHONY: help sync test test-fast lint lint-fix format types types-report docs docs-serve docs-deploy build check clean

UV ?= uv
RUN := $(UV) run
SRC := src/mesoscopy

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

sync:  ## Install/sync the development environment
	$(UV) sync

test:  ## Run the test suite with coverage (term + lcov.info + coverage.xml)
	$(RUN) pytest --cov-config=pyproject.toml --cov=mesoscopy --cov=tests \
		--cov-report=term --cov-report=lcov:lcov.info --cov-report=xml:coverage.xml

test-fast:  ## Run the test suite without coverage
	$(RUN) pytest --no-cov

lint:  ## Lint with ruff
	$(RUN) ruff check $(SRC)

lint-fix:  ## Lint with ruff, applying fixes
	$(RUN) ruff check --fix $(SRC)

format: lint-fix  ## Apply ruff fixes, then format
	$(RUN) ruff format $(SRC)

types:  ## Type-check with mypy
	$(RUN) mypy --install-types --non-interactive $(SRC) tests

types-report:  ## Type-check and write an HTML report to mypy-report/
	$(RUN) mypy --install-types --non-interactive --html-report mypy-report/ $(SRC) tests

docs:  ## Build the documentation (strict)
	$(RUN) mkdocs build --clean --strict

docs-serve:  ## Serve the documentation at localhost:8000
	$(RUN) mkdocs serve --dev-addr localhost:8000

docs-deploy:  ## Deploy the documentation to GitHub Pages
	$(RUN) mkdocs gh-deploy --force

build:  ## Build the sdist and wheel
	$(UV) build

check: lint types test  ## Run lint, type checks and tests

clean:  ## Remove build, coverage and tool cache artefacts
	rm -rf dist/ site/ mypy-report/ .coverage .coverage.* coverage.xml lcov.info
	rm -rf .pytest_cache/ .ruff_cache/ .mypy_cache/
