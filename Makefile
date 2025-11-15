.PHONY: help setup install install-llama run test clean chat chat-mtv web web-mtv web-test memory-stats memory-clear memory-load memory-store memory-retrieve memory-list

PYTHON := python3.12
VENV := .venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python
LLAMA_CPP := llama.cpp

.DEFAULT_GOAL := help

help: ## Show available commands
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make \033[36m<target>\033[0m\n\n"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Create virtual environment
	@$(PYTHON) -m venv $(VENV)

install: setup ## Install dependencies (includes setup)
	@$(PIP) install --upgrade pip -q
	@$(PIP) install -r requirements.txt

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

run: ## Run model converter (MODEL=... QUANTIZE=... OUTPUT_DIR=... TOKEN=... DOWNLOAD_ONLY=true VERBOSE=true)
	@[ -f "$(PYTHON_VENV)" ] || (echo "Run 'make install' first" && exit 1)
	@OPTS=""; \
	[ -n "$(MODEL)" ] && OPTS="$$OPTS --model $(MODEL)"; \
	[ -n "$(QUANTIZE)" ] && OPTS="$$OPTS --quantize $(QUANTIZE)"; \
	[ -n "$(OUTPUT_DIR)" ] && OPTS="$$OPTS --output-dir $(OUTPUT_DIR)"; \
	[ -n "$(TOKEN)" ] && OPTS="$$OPTS --token $(TOKEN)"; \
	[ "$(DOWNLOAD_ONLY)" = "true" ] && OPTS="$$OPTS --download-only"; \
	[ "$(VERBOSE)" = "true" ] && OPTS="$$OPTS --verbose"; \
	$(PYTHON_VENV) bin/model_converter.py $$OPTS

test: ## Test GGUF model (MODEL=... PROMPT=... MAX_TOKENS=... OUTPUT=results.json)
	@[ -f "$(PYTHON_VENV)" ] || (echo "Run 'make install' first" && exit 1)
	@OPTS=""; \
	[ -n "$(MODEL)" ] && OPTS="$$OPTS --model $(MODEL)"; \
	[ -n "$(PROMPT)" ] && OPTS="$$OPTS --prompt \"$(PROMPT)\""; \
	[ -n "$(MAX_TOKENS)" ] && OPTS="$$OPTS --max-tokens $(MAX_TOKENS)"; \
	[ -n "$(OUTPUT)" ] && OPTS="$$OPTS --output $(OUTPUT)"; \
	[ "$(VERBOSE)" = "true" ] && OPTS="$$OPTS --verbose"; \
	$(PYTHON_VENV) scripts/test_model.py $$OPTS

chat: ## Run interactive CLI chat with tools and memory (MODEL=q4_0|q4_1|q8 CONTEXT=16384)
	@[ -f "$(PYTHON_VENV)" ] || (echo "Run 'make install' first" && exit 1)
	@MODEL_TYPE=$${MODEL:-auto}; \
	CONTEXT_SIZE=$${CONTEXT:-16384}; \
	if [ "$$MODEL_TYPE" = "auto" ]; then \
		$(PYTHON_VENV) bin/chat_app.py --context $$CONTEXT_SIZE; \
	else \
		if [ "$$MODEL_TYPE" = "q8" ]; then \
			MODEL_FILE=$$(ls models/*/gguf/*-q8_0.gguf 2>/dev/null | head -1); \
		elif [ "$$MODEL_TYPE" = "q4_1" ]; then \
			MODEL_FILE=$$(ls models/*/gguf/*-q4_1.gguf 2>/dev/null | head -1); \
		else \
			MODEL_FILE=$$(ls models/*/gguf/*-q4_0.gguf 2>/dev/null | head -1); \
		fi; \
		if [ -z "$$MODEL_FILE" ]; then \
			echo "No $$MODEL_TYPE model found. Run 'make run' to convert one."; \
			exit 1; \
		fi; \
		$(PYTHON_VENV) bin/chat_app.py --model "$$MODEL_FILE" --context $$CONTEXT_SIZE; \
	fi

chat-mtv: ## Run CLI chat with MTV MCP server and system prompt (MODEL=q4_0|q4_1|q8 CONTEXT=16384)
	@[ -f "$(PYTHON_VENV)" ] || (echo "Run 'make install' first" && exit 1)
	@MODEL_TYPE=$${MODEL:-auto}; \
	CONTEXT_SIZE=$${CONTEXT:-16384}; \
	if [ "$$MODEL_TYPE" = "auto" ]; then \
		$(PYTHON_VENV) bin/chat_app.py --context $$CONTEXT_SIZE \
			--mcp-server "mtv:kubectl-mtv mcp-server" \
			--system-prompt @docs/mtv-system-prompt.txt; \
	else \
		if [ "$$MODEL_TYPE" = "q8" ]; then \
			MODEL_FILE=$$(ls models/*/gguf/*-q8_0.gguf 2>/dev/null | head -1); \
		elif [ "$$MODEL_TYPE" = "q4_1" ]; then \
			MODEL_FILE=$$(ls models/*/gguf/*-q4_1.gguf 2>/dev/null | head -1); \
		else \
			MODEL_FILE=$$(ls models/*/gguf/*-q4_0.gguf 2>/dev/null | head -1); \
		fi; \
		if [ -z "$$MODEL_FILE" ]; then \
			echo "No $$MODEL_TYPE model found. Run 'make run' to convert one."; \
			exit 1; \
		fi; \
		$(PYTHON_VENV) bin/chat_app.py --model "$$MODEL_FILE" --context $$CONTEXT_SIZE \
			--mcp-server "mtv:kubectl-mtv mcp-server" \
			--system-prompt @docs/mtv-system-prompt.txt; \
	fi

web: ## Run web chat server (MODEL=q4_0|q4_1|q8 CONTEXT=16384 HOST=127.0.0.1 PORT=8000)
	@[ -f "$(PYTHON_VENV)" ] || (echo "Run 'make install' first" && exit 1)
	@MODEL_TYPE=$${MODEL:-auto}; \
	CONTEXT_SIZE=$${CONTEXT:-16384}; \
	HOST_ADDR=$${HOST:-127.0.0.1}; \
	PORT_NUM=$${PORT:-8000}; \
	if [ "$$MODEL_TYPE" = "auto" ]; then \
		echo "Starting web server at http://$$HOST_ADDR:$$PORT_NUM"; \
		$(PYTHON_VENV) bin/web_app.py --host $$HOST_ADDR --port $$PORT_NUM --context $$CONTEXT_SIZE; \
	else \
		if [ "$$MODEL_TYPE" = "q8" ]; then \
			MODEL_FILE=$$(ls models/*/gguf/*-q8_0.gguf 2>/dev/null | head -1); \
		elif [ "$$MODEL_TYPE" = "q4_1" ]; then \
			MODEL_FILE=$$(ls models/*/gguf/*-q4_1.gguf 2>/dev/null | head -1); \
		else \
			MODEL_FILE=$$(ls models/*/gguf/*-q4_0.gguf 2>/dev/null | head -1); \
		fi; \
		if [ -z "$$MODEL_FILE" ]; then \
			echo "No $$MODEL_TYPE model found. Run 'make run' to convert one."; \
			exit 1; \
		fi; \
		echo "Starting web server at http://$$HOST_ADDR:$$PORT_NUM"; \
		$(PYTHON_VENV) bin/web_app.py --model "$$MODEL_FILE" --host $$HOST_ADDR --port $$PORT_NUM --context $$CONTEXT_SIZE; \
	fi

web-mtv: ## Run web server with MTV MCP server and system prompt (MODEL=q4_0|q4_1|q8 CONTEXT=16384 HOST=127.0.0.1 PORT=8000)
	@[ -f "$(PYTHON_VENV)" ] || (echo "Run 'make install' first" && exit 1)
	@MODEL_TYPE=$${MODEL:-auto}; \
	CONTEXT_SIZE=$${CONTEXT:-16384}; \
	HOST_ADDR=$${HOST:-127.0.0.1}; \
	PORT_NUM=$${PORT:-8000}; \
	if [ "$$MODEL_TYPE" = "auto" ]; then \
		echo "Starting MTV web server at http://$$HOST_ADDR:$$PORT_NUM"; \
		$(PYTHON_VENV) bin/web_app.py --host $$HOST_ADDR --port $$PORT_NUM --context $$CONTEXT_SIZE \
			--mcp-server "mtv:kubectl-mtv mcp-server" \
			--system-prompt @docs/mtv-system-prompt.txt; \
	else \
		if [ "$$MODEL_TYPE" = "q8" ]; then \
			MODEL_FILE=$$(ls models/*/gguf/*-q8_0.gguf 2>/dev/null | head -1); \
		elif [ "$$MODEL_TYPE" = "q4_1" ]; then \
			MODEL_FILE=$$(ls models/*/gguf/*-q4_1.gguf 2>/dev/null | head -1); \
		else \
			MODEL_FILE=$$(ls models/*/gguf/*-q4_0.gguf 2>/dev/null | head -1); \
		fi; \
		if [ -z "$$MODEL_FILE" ]; then \
			echo "No $$MODEL_TYPE model found. Run 'make run' to convert one."; \
			exit 1; \
		fi; \
		echo "Starting MTV web server at http://$$HOST_ADDR:$$PORT_NUM"; \
		$(PYTHON_VENV) bin/web_app.py --model "$$MODEL_FILE" --host $$HOST_ADDR --port $$PORT_NUM --context $$CONTEXT_SIZE \
			--mcp-server "mtv:kubectl-mtv mcp-server" \
			--system-prompt @docs/mtv-system-prompt.txt; \
	fi

web-test: ## Test web client API (requires web server to be running)
	@[ -f "$(PYTHON_VENV)" ] || (echo "Run 'make install' first" && exit 1)
	@$(PYTHON_VENV) bin/test_web_client.py

memory-stats: ## Show memory statistics
	@[ -f "$(PYTHON_VENV)" ] || (echo "Run 'make install' first" && exit 1)
	@$(PYTHON_VENV) bin/memory_cli.py stats

memory-clear: ## Clear all stored memories
	@[ -f "$(PYTHON_VENV)" ] || (echo "Run 'make install' first" && exit 1)
	@$(PYTHON_VENV) bin/memory_cli.py clear --force

memory-load: ## Load markdown docs into memory (DOCS=path/to/docs)
	@[ -f "$(PYTHON_VENV)" ] || (echo "Run 'make install' first" && exit 1)
	@if [ -z "$(DOCS)" ]; then \
		echo "Usage: make memory-load DOCS=path/to/docs"; \
		exit 1; \
	fi; \
	$(PYTHON_VENV) bin/memory_cli.py load-docs $(DOCS)

memory-store: ## Store text in memory (TEXT="your text")
	@[ -f "$(PYTHON_VENV)" ] || (echo "Run 'make install' first" && exit 1)
	@if [ -z "$(TEXT)" ]; then \
		echo "Usage: make memory-store TEXT=\"your text here\""; \
		exit 1; \
	fi; \
	$(PYTHON_VENV) bin/memory_cli.py store "$(TEXT)"

memory-retrieve: ## Retrieve memories (QUERY="your query" TOP_K=3)
	@[ -f "$(PYTHON_VENV)" ] || (echo "Run 'make install' first" && exit 1)
	@if [ -z "$(QUERY)" ]; then \
		echo "Usage: make memory-retrieve QUERY=\"your query\""; \
		exit 1; \
	fi; \
	$(PYTHON_VENV) bin/memory_cli.py retrieve "$(QUERY)" --top-k $${TOP_K:-3}

memory-list: ## List stored chunks (FILE=filename SOURCE=source PREVIEW=200 GROUP=true)
	@[ -f "$(PYTHON_VENV)" ] || (echo "Run 'make install' first" && exit 1)
	@CMD="$(PYTHON_VENV) bin/memory_cli.py list"; \
	[ -n "$(FILE)" ] && CMD="$$CMD --file $(FILE)"; \
	[ -n "$(SOURCE)" ] && CMD="$$CMD --source $(SOURCE)"; \
	[ -n "$(PREVIEW)" ] && CMD="$$CMD --preview $(PREVIEW)"; \
	[ "$(GROUP)" = "true" ] && CMD="$$CMD --group-by-source"; \
	eval $$CMD

clean: ## Remove venv and models
	@rm -rf $(VENV) models

