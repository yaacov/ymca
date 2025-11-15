# Web Interface Documentation

The YMCA Web Interface provides a modern, browser-based chat experience with support for tools, memory, and MCP server integration.

## Overview

The web interface offers:
- **Interactive Chat**: Real-time conversational AI with streaming responses
- **Tool Integration**: Automatic tool selection and execution
- **Memory Support**: Access to semantic memory across sessions
- **MCP Servers**: Connect to multiple Model Context Protocol servers
- **Planning Mode**: Multi-step reasoning for complex tasks
- **Customizable**: Configure model parameters, system prompts, and behavior

## Quick Start

### Basic Usage

Start the web server:

```bash
ymca-web
```

Open your browser to: http://localhost:8000

### With MCP Server

Start with MCP server integration:

```bash
ymca-web --mcp-server "mtv:kubectl-mtv mcp-server"
```

### Custom Configuration

Start with custom settings:

```bash
ymca-web \
  --model path/to/model.gguf \
  --port 8080 \
  --system-prompt @custom-prompt.txt \
  --enable-memory \
  --context 32768
```

## Installation

The web interface is included with YMCA.

### Setup Virtual Environment

Create and activate a virtual environment:

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

# Or with development tools
pip install -e ".[dev]"
```

## Command-Line Options

### Model Configuration

```bash
ymca-web --model PATH            # Path to GGUF model file
ymca-web --context SIZE          # Context window size in tokens (default: 32768)
ymca-web --gpu-layers N          # Number of layers to offload to GPU (default: -1 = all)
ymca-web --threads N             # Number of CPU threads (default: auto)
```

### Server Configuration

```bash
ymca-web --host HOST             # Server host (default: localhost)
ymca-web --port PORT             # Server port (default: 8000)
ymca-web --reload                # Enable auto-reload for development
```

### Feature Flags

```bash
ymca-web --enable-memory         # Enable semantic memory tool
ymca-web --memory-dir DIR        # Memory storage directory (default: data/tools/memory)
ymca-web --enable-planning       # Enable multi-step planning (default: enabled)
ymca-web --enable-tool-selector  # Enable intelligent tool selection (default: enabled)
ymca-web --max-tool-selector N   # Max tools to select (default: 3)
```

### System Prompt

```bash
ymca-web --system-prompt TEXT    # Direct system prompt text
ymca-web --system-prompt @FILE   # Load system prompt from file
```

### MCP Integration

```bash
ymca-web --mcp-server "name:command arg1 arg2"
```

Multiple servers:
```bash
ymca-web \
  --mcp-server "mtv:kubectl-mtv mcp-server" \
  --mcp-server "files:mcp-server-files /path/to/files"
```

### Logging

```bash
ymca-web --verbose               # Enable verbose logging
ymca-web --log-file FILE         # Write logs to file
```

## API Endpoints

The web interface exposes REST endpoints for programmatic access.

### POST /chat

Send a chat message and receive a response.

**Request:**
```json
{
  "message": "What is MCP?",
  "enable_tools": true,
  "enable_planning": true,
  "max_iterations": 5,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "response": "MCP stands for Model Context Protocol...",
  "status": "success"
}
```

**Parameters:**
- `message` (string, required): User message
- `enable_tools` (boolean, default: true): Enable tool usage
- `enable_planning` (boolean, default: true): Enable multi-step planning
- `max_iterations` (int, default: 5): Maximum planning iterations
- `temperature` (float, default: 0.7): Sampling temperature (0.0-2.0)

**Example using curl:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is MCP?", "temperature": 0.5}'
```

**Example using Python:**
```python
import requests

response = requests.post("http://localhost:8000/chat", json={
    "message": "What is MCP?",
    "enable_tools": True,
    "temperature": 0.7
})

print(response.json()["response"])
```

### POST /chat/stream

Stream chat responses for real-time display.

**Request:**
Same as `/chat` endpoint

**Response:**
Server-Sent Events (SSE) stream with incremental tokens

**Example:**
```bash
curl -N http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about YMCA"}'
```

### GET /health

Check server health status.

**Response:**
```json
{
  "status": "healthy",
  "message": "Server is running"
}
```

### GET /

Serve the web interface HTML page.

## Web UI Features

### Chat Interface

The web UI provides:
- **Message History**: View conversation history
- **Streaming Responses**: See responses as they're generated
- **Tool Indicators**: Visual feedback when tools are used
- **Error Handling**: Clear error messages and recovery

### Configuration Panel

Adjust settings in the web interface:
- Temperature control
- Enable/disable tools
- Enable/disable planning
- Max iterations for planning

### Keyboard Shortcuts

- `Enter`: Send message
- `Shift + Enter`: New line in message
- `Ctrl/Cmd + K`: Clear conversation
- `Ctrl/Cmd + L`: Focus message input

## Deployment

### Development Mode

For local development with auto-reload:

```bash
ymca-web --reload --verbose
```
