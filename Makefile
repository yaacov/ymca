.PHONY: help setup install-llama convert-model run web memory-load clean

PYTHON := python3.12
VENV := .venv
PYTHON_VENV := $(VENV)/bin/python
LLAMA_CPP := llama.cpp

.DEFAULT_GOAL := help

help: ## Show available commands
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make \033[36m<target>\033[0m\n\n"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Create virtual environment and install dependencies
	@$(PYTHON) -m venv $(VENV)
	@$(VENV)/bin/pip install --upgrade pip -q
	@$(VENV)/bin/pip install -e . -q
	@echo "✓ Virtual environment created and dependencies installed"

install-llama: ## Install llama.cpp for GGUF conversion
	@if [ -d "$(LLAMA_CPP)" ]; then \
		echo "llama.cpp already exists, pulling latest changes..."; \
		cd $(LLAMA_CPP) && git pull; \
	else \
		echo "Cloning llama.cpp..."; \
		git clone https://github.com/ggerganov/llama.cpp $(LLAMA_CPP); \
	fi
	@echo "Building llama.cpp with CMake..."
	@cd $(LLAMA_CPP) && mkdir -p build && cd build && cmake .. && cmake --build . --config Release
	@echo "✓ llama.cpp installed successfully"

convert-model: ## Convert/download models (MODEL=... QUANTIZE=... OUTPUT_DIR=... TOKEN=... DOWNLOAD_ONLY=true VERBOSE=true)
	@[ -f "$(PYTHON_VENV)" ] || (echo "Run 'make setup' first" && exit 1)
	@OPTS=""; \
	[ -n "$(MODEL)" ] && OPTS="$$OPTS --model $(MODEL)"; \
	[ -n "$(QUANTIZE)" ] && OPTS="$$OPTS --quantize $(QUANTIZE)"; \
	[ -n "$(OUTPUT_DIR)" ] && OPTS="$$OPTS --output-dir $(OUTPUT_DIR)"; \
	[ -n "$(TOKEN)" ] && OPTS="$$OPTS --token $(TOKEN)"; \
	[ "$(DOWNLOAD_ONLY)" = "true" ] && OPTS="$$OPTS --download-only"; \
	[ "$(VERBOSE)" = "true" ] && OPTS="$$OPTS --verbose"; \
	$(PYTHON_VENV) bin/model_converter.py $$OPTS

run: ## Run interactive CLI chat with tools and memory
	@[ -f "$(PYTHON_VENV)" ] || (echo "Run 'make setup' first" && exit 1)
	@$(PYTHON_VENV) bin/chat_app.py

web: ## Run web chat server (HOST=127.0.0.1 PORT=8000)
	@[ -f "$(PYTHON_VENV)" ] || (echo "Run 'make setup' first" && exit 1)
	@HOST_ADDR=$${HOST:-127.0.0.1}; \
	PORT_NUM=$${PORT:-8000}; \
	echo "Starting web server at http://$$HOST_ADDR:$$PORT_NUM"; \
	$(PYTHON_VENV) bin/web_app.py --host $$HOST_ADDR --port $$PORT_NUM

memory-load: ## Load markdown docs into memory (DOCS=path/to/docs)
	@[ -f "$(PYTHON_VENV)" ] || (echo "Run 'make setup' first" && exit 1)
	@if [ -z "$(DOCS)" ]; then \
		echo "Usage: make memory-load DOCS=path/to/docs"; \
		exit 1; \
	fi; \
	$(PYTHON_VENV) bin/memory_cli.py load-docs $(DOCS)

clean: ## Remove venv and models
	@rm -rf $(VENV) models

