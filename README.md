# YMCA - Yaacov's MCP Agent

YMCA (Yaacov's MCP Agent) is a local-first AI agent framework designed to run entirely on your own hardware. It leverages the Model Context Protocol (MCP) to connect to various tool servers while using llama.cpp with aggressive quantization for efficient local inference. This approach ensures privacy, reduces costs, and eliminates dependencies on cloud services.

## Overview

YMCA combines the power of large language models with the flexibility of MCP servers, all while maintaining a small footprint through quantized models. The framework includes built-in memory capabilities, allowing the agent to maintain context across conversations using semantic search and vector embeddings.

## Key Features

### Local Execution
Run AI models directly on your hardware without sending data to external servers. YMCA uses llama.cpp for fast, efficient inference on both CPU and GPU, supporting a wide range of quantized model formats.

### MCP Server Integration
Connect to Model Context Protocol servers to extend your agent's capabilities. MCP provides a standardized way to integrate tools, data sources, and services, allowing your agent to interact with databases, APIs, file systems, and more.

### Aggressive Quantization
Support for heavily quantized models (4-bit, 5-bit, 8-bit) enables running capable models on modest hardware. This makes it possible to deploy AI agents on laptops, workstations, or edge devices without requiring expensive GPU infrastructure.

### Semantic Memory
Built-in memory system using ChromaDB for vector storage and sentence transformers for embeddings. The agent can remember and recall information from past conversations, creating a persistent context that improves over time.

### Privacy-Focused
All processing happens locally. Your conversations, data, and model interactions never leave your machine, ensuring complete privacy and data sovereignty.

## Installation

Install YMCA in development mode to get started:

```bash
pip install -e .
```

For development with testing and linting tools:

```bash
pip install -e ".[dev]"
```

## Quick Start

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

### Memory Management
Interact with the agent's memory system:

```bash
ymca-memory
```

Use this to query, add, or manage stored memories and embeddings.

### Model Conversion
Convert Hugging Face models to GGUF format for use with llama.cpp:

```bash
ymca-convert
```

### MCP Server Integration
Connect to MCP servers and use custom system prompts:

```bash
# Use with MCP server and specialized system prompt
ymca-chat --mcp-server "mtv:kubectl-mtv mcp-server" --system-prompt @docs/mtv-system-prompt.txt

# Multiple MCP servers
ymca-chat --mcp-server "server1:cmd1" --mcp-server "server2:cmd2"

# Or use convenient make targets
make chat-mtv   # CLI chat with MTV MCP server
make web-mtv    # Web interface with MTV MCP server
```

See `docs/mtv-mcp-setup.md` for detailed MTV MCP server configuration.

## Architecture

YMCA consists of several core components:

- **Model Handler**: Manages loading and inference with quantized models via llama.cpp
- **Chat API**: Provides the conversational interface and manages dialog flow
- **Memory Tool**: Implements semantic memory using vector embeddings and ChromaDB
- **Tool Selector**: Intelligently chooses and invokes MCP tools based on context
- **MCP Integration**: Connects to external MCP servers for extended capabilities

## Requirements

- Python 3.9 or higher
- Local hardware (CPU or GPU)
- Sufficient RAM based on model size (typically 4-16GB)

## License

MIT

