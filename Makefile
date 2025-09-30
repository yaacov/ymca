# LLM Chat Application Makefile

VENV_NAME = .venv
PYTHON = python3
PIP = $(VENV_NAME)/bin/pip
PYTHON_VENV = $(VENV_NAME)/bin/python

.PHONY: help
help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

$(VENV_NAME):
	$(PYTHON) -m venv $(VENV_NAME)
	$(PIP) install --upgrade pip

.PHONY: setup
setup: $(VENV_NAME) ## Create virtual environment and install dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt

.PHONY: update
update: $(VENV_NAME) ## Update all dependencies to latest versions
	$(PIP) install --upgrade -r requirements.txt
	$(PIP) install --upgrade -r requirements-dev.txt

.PHONY: run
run: $(VENV_NAME) ## Run the LLM chat application
	$(PYTHON_VENV) main.py

.PHONY: device-info
run-device-info: $(VENV_NAME) ## Show device information and test device-agnostic system
	$(PYTHON_VENV) device_info.py

.PHONY: clean
clean: ## Remove cache files and compiled Python files
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "*.pyd" -delete

.PHONY: clean-cache
clean-cache: ## Remove all cache directories and files
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf .coverage.*
	rm -rf htmlcov/
	rm -rf .tox/
	rm -rf *.egg-info/
	rm -rf .eggs/
	rm -rf build/
	rm -rf dist/
	find . -name ".DS_Store" -delete
	find . -name "*.orig" -delete
	find . -name "*.rej" -delete

.PHONY: clean-models
clean-models: ## Remove downloaded models (WARNING: This will delete all model files)
	@echo "WARNING: This will delete all downloaded models in the models/ directory"
	@echo "Press Ctrl+C to cancel, or Enter to continue..."
	@read dummy
	rm -rf models/

.PHONY: clean-all
clean-all: clean clean-cache ## Remove virtual environment and all cache files
	rm -rf $(VENV_NAME)

.PHONY: clean-everything
clean-everything: clean-all clean-models ## Remove everything including models (use with caution)

.PHONY: lint
lint: $(VENV_NAME) ## Run code linting and formatting checks
	$(VENV_NAME)/bin/flake8 main.py modules/ tests/ device_info.py
	$(VENV_NAME)/bin/mypy main.py modules/ device_info.py
	$(VENV_NAME)/bin/black --check main.py modules/ tests/ device_info.py
	$(VENV_NAME)/bin/isort --check main.py modules/ tests/ device_info.py

.PHONY: format
format: $(VENV_NAME) ## Format code with black and isort
	$(VENV_NAME)/bin/black main.py modules/ tests/ device_info.py
	$(VENV_NAME)/bin/isort main.py modules/ tests/ device_info.py

.PHONY: test
test: $(VENV_NAME) ## Run pytest tests with coverage
	$(VENV_NAME)/bin/pytest tests/ -v --cov=modules --cov-report=term-missing
