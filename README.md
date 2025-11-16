# YMCA - Yaacov's MCp Agent

YMCA (Yaacov's MCp Agent) is a local-first AI agent framework designed to run entirely on your own hardware. It leverages the Model Context Protocol (MCP) to connect to various tool servers while using small language models (SML) with aggressive quantization for efficient local inference. This approach ensures privacy, reduces costs, and eliminates dependencies on cloud services.

<p align="center">
  <img src="docs/monky-wrench.png" width="400" alt="YMCA Logo">
</p>

## Overview

YMCA combines the power of small language models (SML) with the flexibility of MCP servers, all while maintaining a small footprint through quantized models. The framework uses llama.cpp and includes built-in memory capabilities, allowing the agent to maintain context across conversations using semantic search and vector embeddings.

## Key Features

### Local Execution
Run small AI models directly on your hardware without sending data to external servers. YMCA uses llama.cpp for fast, efficient inference on both CPU and GPU, supporting a wide range of quantized model formats.

### MCP Server Integration
Connect to Model Context Protocol servers to extend your agent's capabilities. MCP provides a standardized way to integrate tools, data sources, and services, allowing your agent to interact with databases, APIs, file systems, and more.

### Aggressive Quantization
Support for heavily quantized models (4-bit, 5-bit, 8-bit) enables running capable models on modest hardware. This makes it possible to deploy AI agents on laptops, workstations, or edge devices without requiring expensive GPU infrastructure.

### Semantic Memory
Built-in memory system using ChromaDB for vector storage and sentence transformers for embeddings. The agent can remember and recall information from past conversations, creating a persistent context that improves over time.

### Privacy-Focused
All processing happens locally. Your conversations, data, and model interactions never leave your machine, ensuring complete privacy and data sovereignty.

### Intelligent Tool Selection
YMCA implements **selective context loading** (also known as tool selection) to optimize context window usage. Instead of loading all available tools into the model's context, the system uses semantic search to identify and provide only the most relevant tools for each query. This technique significantly reduces the token count in system prompts, enabling more efficient use of limited context windows.

### Answer Refinement
Automatic post-processing step that improves response clarity and formatting while preserving technical accuracy. Responses are polished for better readability, structure, and presentation. Enabled by default (disable with `--no-refine-answers` flag or uncheck the web UI checkbox).


## Installation

### Create a Virtual Environment

It's recommended to use a virtual environment to avoid conflicts with system packages:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Install YMCA

With the virtual environment activated:

```bash
# Basic installation
pip install -e .

# Or with development tools (testing, linting)
pip install -e ".[dev]"
```

### Hardware-Specific Installation for Optimal Performance

For best performance, install `llama-cpp-python` with hardware-specific acceleration. The basic installation builds for CPU only, but you can enable GPU acceleration and other optimizations by setting `CMAKE_ARGS` before installation:

**Apple Silicon (M1, M2, M3) - Metal Acceleration:**

```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
```

**NVIDIA GPUs - CUDA Acceleration:**

```bash
# Build from source with CUDA support
CMAKE_ARGS="-DGGML_CUDA=on" pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir

# Check your CUDA version first:
nvidia-smi

# Or use pre-built wheels (faster, replace cu121 with your CUDA version)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

**AMD GPUs - ROCm/HIP Acceleration:**

```bash
CMAKE_ARGS="-DGGML_HIPBLAS=on" pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
```

**CPU Optimization - OpenBLAS:**

```bash
CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
```

**Cross-Platform GPU - Vulkan:**

```bash
CMAKE_ARGS="-DGGML_VULKAN=on" pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
```

> **Note:** After installing with hardware acceleration, ensure your virtual environment is activated, then reinstall YMCA: `pip install -e .`

### Chat Interface
Start an interactive chat session with the agent:

```bash
ymca-chat
```

### Web Interface
Launch the web-based chat interface:

```bash
ymca-web
```

The web server will start on `http://localhost:8000` by default.

**[Web Interface Documentation](docs/web-interface.md)** - Complete guide to the web UI, API endpoints, deployment, and customization.

### Memory Management
Interact with the agent's memory system:

```bash
ymca-memory
```

Use this to query, add, or manage stored memories and embeddings.

**[Memory Tool Documentation](docs/memory-tool.md)** - Comprehensive guide to storing, retrieving, and managing semantic memories.

### Model Conversion
Convert Hugging Face models to GGUF format for use with llama.cpp:

```bash
ymca-convert model-name --quantize q4_k_m
```

**[Model Converter Documentation](docs/model-converter.md)** - Complete guide to downloading, converting, and quantizing models.

### MCP Server Integration
Connect to MCP servers and use custom system prompts:

```bash
# Use with MCP server and specialized system prompt
ymca-chat --mcp-server "mtv:kubectl-mtv mcp-server" --system-prompt @docs/mtv-system-prompt.txt

# Multiple MCP servers
ymca-chat --mcp-server "server1:cmd1" --mcp-server "server2:cmd2"
```

See `docs/mtv-mcp-setup.md` for detailed MTV MCP server configuration.


