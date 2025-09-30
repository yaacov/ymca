"""Memory-related tools for comprehensive information storage, retrieval, and management."""

from typing import Any, Callable, Dict, List, Optional, Tuple, cast


def create_memory_tools(memory_manager: Any) -> List[Tuple[Dict[str, Any], Callable[..., Any]]]:
    """Create comprehensive memory management tools for storage, search, retrieval, and organization."""
    tools = []

    # Store memory tool
    store_tool = {
        "name": "store_memory",
        "description": (
            "Store information in long-term memory with automatic chunking, "
            "summary generation, and semantic indexing. Use this to save important "
            "information, research findings, or context that should be remembered "
            "for future conversations. Automatically generates searchable questions "
            "and summaries."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Content to store in memory"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags for categorization and filtering"},
                "url": {"type": "string", "description": "Optional source URL if content came from a webpage"},
                "line_numbers": {"type": "string", "description": "Optional line numbers if content came from a file (e.g., '10-15', '10', '10,12-15')"},
            },
            "required": ["content"],
        },
        "category": "memory",
        "enabled": True,
    }

    def store_handler(content: str, tags: Optional[List[str]] = None, url: Optional[str] = None, line_numbers: Optional[str] = None) -> str:
        memory_id = memory_manager.store_memory(content, tags, url, line_numbers)
        source_info = []
        if url:
            source_info.append(f"URL: {url}")
        if line_numbers:
            source_info.append(f"Lines: {line_numbers}")
        source_str = f" ({', '.join(source_info)})" if source_info else ""
        return f"Stored in memory with ID: {memory_id}{source_str}\nContent length: {len(content)} characters\nTags: {tags or 'none'}"

    # Search memories tool
    search_tool = {
        "name": "search_memories",
        "description": (
            "Search through stored memories using semantic similarity. Use this to "
            "find relevant information from previously stored content, research "
            "findings, or context. Returns ranked results with relevance scores "
            "and matched questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query to find relevant memories"},
                "max_results": {"type": "integer", "description": "Maximum number of results to return", "default": 5},
                "similarity_threshold": {"type": "number", "description": "Minimum similarity score (0.0-1.0)", "default": 0.1},
            },
            "required": ["query"],
        },
        "category": "memory",
        "enabled": True,
    }

    def search_handler(query: str, max_results: int = 5, similarity_threshold: float = 0.1) -> str:
        results = memory_manager.search_memories(query, max_results, similarity_threshold)
        if not results:
            return f"No memories found for query: {query}"

        formatted = [f"Memory Search Results for: {query}\n"]
        for i, result in enumerate(results, 1):
            memory = result.memory
            formatted.append(f"{i}. Memory ID: {memory.id}")
            formatted.append(f"   Relevance: {result.relevance_score:.3f}")
            formatted.append(f"   Summary: {memory.summary}")
            formatted.append(f"   Tags: {', '.join(memory.tags) if memory.tags else 'none'}")
            if memory.url:
                formatted.append(f"   Source: {memory.url}")
            if memory.line_numbers:
                formatted.append(f"   Lines: {memory.line_numbers}")
            formatted.append(f"   Created: {memory.created_at}")
            formatted.append(f"   Matched questions: {', '.join(result.matched_questions)}")

            # Show preview of content
            content_preview = memory.content[:200] + "..." if len(memory.content) > 200 else memory.content
            formatted.append(f"   Content preview: {content_preview}")
            formatted.append("")

        return "\n".join(formatted)

    # Register all tools
    tools.extend(
        [
            (store_tool, cast(Callable[..., Any], store_handler)),
            (search_tool, cast(Callable[..., Any], search_handler)),
        ]
    )

    return tools
