SHELL        := /usr/bin/env bash
MAKEFLAGS    += --silent

-include .env
export

SRC_DIR      := src
TEST_DIR     := tests
CHECK_DIRS   := $(SRC_DIR) $(TEST_DIR)
PYTEST_FLAGS := --no-header --capture=no

TEX_DIR      := tex
REPORT_FILE  := report

all: test run clean compile wordcount clean

.PHONY: help
help: ## Show the available commands
	@echo "Available commands:"
	@grep -h -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: format
format: ## Format code
	black $(CHECK_DIRS)
	isort $(CHECK_DIRS)

.PHONY: check
check: format-check lint type-check test ## Launch all checks

.PHONY: format-check
format-check: ## Check the code format
	black --check $(CHECK_DIRS)
	isort --check $(CHECK_DIRS)

.PHONY: lint
lint: ## Lint
	ruff check $(CHECK_DIRS)

.PHONY: lint-fix
lint-fix: ## Lint, and auto-fix the issues where possible
	ruff check --fix $(CHECK_DIRS)

.PHONY: type-check
type-check: ## Check types
	mypy $(CHECK_DIRS)

.PHONY: test
test: ## Run tests
	pytest $(PYTEST_FLAGS) $(TEST_DIR)

.PHONY: run
run: ## Run the main program
	python main.py

.PHONY: clean
clean: ## Clean the repository
	rm -rf dist
	rm -rf .coverage
	rm -rf *.egg-info
	find . -type f -name *.DS_Store -ls -delete
	find . | grep -E '(__pycache__|\.pyc|\.pyo)' | xargs rm -rf
	find . | grep -E .mypy_cache | xargs rm -rf
	find . | grep -E .pytest_cache | xargs rm -rf
	find . | grep -E .ruff_cache | xargs rm -rf
	find . | grep -E .ipynb_checkpoints | xargs rm -rf
	find . | grep -E .trash | xargs rm -rf
	find . -type f \( -name '*.aux' \
		-o -name '*.snm' \
		-o -name '*.nav' \
		-o -name '*.lof' \
		-o -name '*.log' \
		-o -name '*.lot' \
		-o -name '*.fls' \
		-o -name '*.out' \
		-o -name '*.toc' \
		-o -name '*.fmt' \
		-o -name '*.fot' \
		-o -name '*.cb' \
		-o -name '*.cb2' \
		-o -name '.*lb' \
		-o -name '*.run.xml' \
		-o -name '*.bcf' \
		-o -name '*.bbl' \
		-o -name '*.bbl-SAVE-ERROR' \
		-o -name '*.blg' \
		-o -name '*.fdb_latexmk' \
		-o -name '*.synctex.gz' \
		-o -name '*-SAVE-ERROR' \) -delete

.PHONY: compile
compile: ## Compile the LaTeX documents
	cd $(TEX_DIR) && \
		pdflatex $(REPORT_FILE) && \
		biber $(REPORT_FILE) && \
		pdflatex $(REPORT_FILE) && \
		pdflatex $(REPORT_FILE)

.PHONY: wordcount
wordcount: ## Display the wordcount
	cd $(TEX_DIR) && texcount -merge $(REPORT_FILE).tex
