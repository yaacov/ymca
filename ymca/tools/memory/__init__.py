"""
Memory Tool Package - RAG-based memory system with ChromaDB.

Main Components:
- MemoryTool: Main interface for storing and retrieving memories
- ChunkStorage: File-based storage for text chunks (one file per chunk)
- VectorStore: ChromaDB-based vector storage for embeddings
- TextChunker: Text chunking with overlap
- Embedder: Sentence transformer embeddings
- MemoryRetriever: Search and retrieval logic

Usage:
    from ymca.tools.memory import MemoryTool
    from ymca.core.model_handler import ModelHandler
    
    handler = ModelHandler(model_path="path/to/model.gguf")
    memory = MemoryTool(model_handler=handler)
    
    # Store memory
    memory.store_memory("Some text to remember", source="user")
    
    # Retrieve memory
    results = memory.retrieve_memory("search query")
"""

from .tool import MemoryTool
from .chunker import TextChunker
from .embedder import Embedder
from .storage import ChunkStorage
from .vector_store import VectorStore
from .retriever import MemoryRetriever

__all__ = [
    "MemoryTool",
    "TextChunker",
    "Embedder",
    "ChunkStorage",
    "VectorStore",
    "MemoryRetriever",
]

