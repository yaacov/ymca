# Memory Tool Documentation

The Memory Tool provides persistent semantic memory for YMCA, allowing the agent to store, retrieve, and manage information across conversations using vector embeddings and ChromaDB.

## Overview

The memory tool enables:
- **Semantic Storage**: Store text with automatic chunking and embedding generation
- **Intelligent Retrieval**: Search memories using natural language queries with similarity matching
- **Document Loading**: Bulk import markdown documentation with automatic processing
- **Memory Management**: List, inspect, and clear stored memories

## Architecture

The memory system uses a two-stage approach:

1. **Storage**: Large chunks (default 4000 characters) stored as text files for efficient retrieval
2. **Embeddings**: Smaller sub-chunks (1200 characters) embedded with generated questions for semantic search

This hybrid approach balances storage efficiency with search accuracy, enabling fast retrieval of relevant context.

## Installation

The memory tool is included with YMCA.

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
pip install -e .
```

## Quick Start

### Basic Usage

Store information:
```bash
ymca-memory store "Python was created by Guido van Rossum in 1991"
```

Retrieve information:
```bash
ymca-memory retrieve "Who created Python?"
```

### Loading Documentation

Load markdown files from a directory:
```bash
ymca-memory load-docs /path/to/docs
```

Load with custom patterns:
```bash
ymca-memory load-docs /path/to/docs --pattern "*.md"
```

### Memory Management

View statistics:
```bash
ymca-memory stats
```

List all stored chunks:
```bash
ymca-memory list
```

List with preview:
```bash
ymca-memory list --preview 200
```

Filter by source:
```bash
ymca-memory list --file README.md
```

Group by source:
```bash
ymca-memory list --group-by-source
```

Clear all memories:
```bash
ymca-memory clear --force
```

## Configuration

### Chunking Strategy

The memory tool uses two types of chunks:

**Storage Chunks (Large)**
- Default size: 4000 characters
- Default overlap: 400 characters (10%)
- Purpose: Efficient storage and retrieval of complete context
- Stored as text files in `data/tools/memory/chunks/`

**Embedding Chunks (Small)**
- Fixed size: 1200 characters
- Purpose: Semantic search with vector embeddings
- Enhanced with 2 generated questions per chunk
- Stored in ChromaDB vector database

### Question Generation

To improve semantic search accuracy, the memory tool automatically generates questions for each embedding chunk:

- 2 questions per 1200-character sub-chunk
- Questions simulate potential user queries
- Embedded alongside the chunk text
- Improves retrieval relevance for conversational queries

### Storage Structure

```
data/tools/memory/
├── chunks/           # Text storage
│   ├── chunk_0.txt
│   ├── chunk_1.txt
│   └── ...
├── vectors/          # Vector embeddings
│   └── chroma/       # ChromaDB database
└── metadata.json     # Storage metadata
```
