# Tool Index Preservation

## Overview

The tool selector now preserves the `tool_index.json` file across restarts and reuses existing tool queries instead of regenerating them every time. This improves startup performance and maintains consistency.

## Problem

Previously, the tool selector would:
1. Clear all tool data on startup
2. Regenerate example queries for every tool on every startup
3. Waste time and tokens regenerating identical queries
4. Potentially generate different queries each time, affecting consistency

## Solution

The tool selector now:
1. **Loads existing index on startup** - Reads `tool_index.json` if it exists
2. **Preserves existing queries** - Uses loaded queries instead of regenerating
3. **Only generates for new tools** - Only calls LLM for tools not in the index
4. **Maintains consistency** - Same queries across restarts

## Implementation

### 1. Load Existing Index on Initialization

```python
def __init__(self, ...):
    # ... other initialization ...
    
    # Load existing tool index if available
    self._load_existing_index()
```

### 2. Load Index Method

```python
def _load_existing_index(self):
    """Load existing tool index from disk if available."""
    index_file = self.cache_dir / "tool_index.json"
    if not index_file.exists():
        return
    
    with open(index_file, 'r', encoding='utf-8') as f:
        index_data = json.load(f)
    
    tools_data = index_data.get("tools", {})
    for tool_name, tool_data in tools_data.items():
        example_queries = tool_data.get("example_queries", [])
        description = tool_data.get("description", "")
        
        if example_queries:
            self.tool_queries[tool_name] = example_queries
        if description:
            self.tool_descriptions[tool_name] = description
```

### 3. Check Before Generating

```python
def index_tools(self, tools):
    for name, tool in tools.items():
        # Check if we already have queries for this tool
        if name in self.tool_queries and self.tool_queries[name]:
            example_queries = self.tool_queries[name]
            logger.debug(f"Using existing queries for tool '{name}'")
        else:
            # Generate new queries only for new tools
            example_queries = self._generate_example_queries(name, tool.description)
            logger.debug(f"Generated new queries for tool '{name}'")
```

## Benefits

### 1. Performance
- **Faster startup**: No LLM calls for existing tools
- **Reduced token usage**: Only generate queries once per tool
- **Immediate availability**: Tools ready as soon as index is loaded

### 2. Consistency
- **Same queries across restarts**: No variation in tool matching
- **Reproducible behavior**: Debugging is easier with consistent queries
- **Stable tool selection**: Same user queries match same tools

### 3. Efficiency
- **Incremental updates**: Only new tools require generation
- **Cached embeddings**: Queries are re-embedded but not regenerated
- **Manual override possible**: Edit `tool_index.json` to customize queries

## Logging

The tool selector now provides detailed logging:

```
INFO: Loaded existing tool index with 1 tools from data/tools/selector/tool_index.json
INFO: Indexing 1 tools for semantic selection
DEBUG: Using existing 5 queries for tool 'retrieve_memory'
INFO: Indexed 1 tools: 1 existing, 0 new
INFO: Saved tool index to data/tools/selector/tool_index.json (queries only, no embeddings)
```

Key indicators:
- **"Loaded existing tool index"**: Index file found and loaded
- **"Using existing N queries"**: Tool reusing cached queries
- **"Generated N new queries"**: New tool, queries generated
- **"X existing, Y new"**: Summary of tool indexing

## File Structure

### tool_index.json

```json
{
  "metadata": {
    "num_tools": 1,
    "embedding_dim": 384,
    "note": "Embeddings are stored in memory only, not persisted to disk"
  },
  "tools": {
    "retrieve_memory": {
      "description": "Search documentation and knowledge base...",
      "example_queries": [
        "How do I configure authentication?",
        "What are the migration types?",
        "How to troubleshoot connection errors?"
      ],
      "query_stats": [
        {"query": "How do I configure authentication?", "embedding_norm": 1.0},
        {"query": "What are the migration types?", "embedding_norm": 1.0},
        {"query": "How to troubleshoot connection errors?", "embedding_norm": 1.0}
      ]
    }
  }
}
```

**Note**: Embeddings are NOT stored in the file (too large). They are regenerated on startup from the stored queries, which is fast.

## Customization

You can manually edit `tool_index.json` to customize tool queries:

1. **Edit queries**: Change `example_queries` to better match expected user queries
2. **Add queries**: Increase coverage by adding more examples
3. **Remove queries**: Reduce noise by removing poor matches
4. **Reset tool**: Delete tool entry to force regeneration on next startup

After editing, restart the application to reload the index.

## Migration from Old Behavior

If you want to regenerate all queries:

1. Delete or rename `data/tools/selector/tool_index.json`
2. Restart the application
3. New queries will be generated for all tools
4. New index will be saved

## Files Modified

- `ymca/chat/tool_selector.py`:
  - Added `_load_existing_index()` method
  - Modified `__init__()` to call loader
  - Modified `index_tools()` to check for existing queries
  - Enhanced logging for transparency

## Testing

See the test output demonstrating:
1. Index file preservation across restarts
2. Existing queries loaded correctly
3. No regeneration for existing tools
4. Incremental addition of new tools

## Summary

The tool index is now:
- ✅ **Preserved** across restarts (not cleared)
- ✅ **Reused** for existing tools (not regenerated)
- ✅ **Incremental** for new tools (only generate what's needed)
- ✅ **Customizable** via manual JSON editing
- ✅ **Efficient** in both time and token usage

