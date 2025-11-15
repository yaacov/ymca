# Memory Retrieval System Improvements

## Summary

Implemented **dual-sided approach** to improve memory retrieval accuracy:

**Retrieval Side (this document):**
1. **LLM-based Query Expansion** - Automatically expand short/vague queries
2. **Better Tool Instructions** - Guide the LLM to formulate better queries

**Storage Side ([see question-generation-improvements.md](question-generation-improvements.md)):**
3. **Enhanced Question Generation** - Generate better questions during chunk storage
4. **Quality Validation** - Ensure generated questions meet quality standards

Together, these improvements create a **complete solution** from storage to retrieval.

## Changes Made

### 1. Query Expansion in Retriever (`ymca/tools/memory/retriever.py`)

**New Method: `expand_query()`**
- Automatically expands short queries (≤10 words) using the LLM
- Converts vague queries into detailed technical questions
- Example: "authentication" → "how to configure authentication with examples and troubleshooting"
- Uses temperature=0.3 for consistent, focused expansions
- Gracefully falls back to original query if expansion fails

**Updated Method: `retrieve()`**
- Added `expand_query` parameter (default: True)
- Expands query before embedding for better semantic matching
- Logs query expansions for debugging

### 2. Improved Tool Instructions (`ymca/tools/memory/tool.py`)

**Enhanced `RETRIEVE_TOOL_DEF`**
- Added detailed query formatting guidance with examples
- Shows GOOD vs BAD query patterns:
  - ✓ GOOD: "how to configure authentication with step by step examples"
  - ✗ BAD: "authentication" (too vague)
- Instructs LLM to include HOW-TO keywords and action verbs
- Increased default `max_results` from 3 → 5 for better coverage

**Updated Methods:**
- `retrieve_memory()`: Now defaults to 5 results and enables query expansion
- `create_retrieve_tool_function()`: Uses new defaults in tool wrapper

### 3. System Prompt Enhancement (Application-Specific)

**Added Query Quality Guidance:**
- Concrete examples of good vs bad queries
- Emphasizes 10-20 word detailed queries
- Shows query transformation examples for context-specific usage
- Note: System prompts are application-specific and can be customized per use case

## How It Works

### Before (Original Behavior)
```
User: "how do I configure authentication?"
  ↓
LLM: retrieve_memory(query="authentication")  ← Too vague!
  ↓
Embedding: [vectors for "authentication"]
  ↓
Results: Might match various auth topics, not specific (3 results)
```

### After (Improved Behavior)

**Path 1: LLM Creates Better Query (Guided by Tool Instructions)**
```
User: "how do I configure authentication?"
  ↓
LLM: retrieve_memory(query="how to configure authentication step by step with examples")  ← Specific!
  ↓
Query Expansion: (already good, skipped)
  ↓
Embedding: [vectors for detailed query]
  ↓
Results: Matches step-by-step configuration guides (5 results)
```

**Path 2: LLM Uses Short Query (Expanded Automatically)**
```
User: "how do I configure authentication?"
  ↓
LLM: retrieve_memory(query="authentication")
  ↓
Query Expansion: "how to configure authentication with detailed examples and troubleshooting"  ← Auto-expanded!
  ↓
Embedding: [vectors for expanded query]
  ↓
Results: Better matches (5 results)
```

## Configuration Options

### Disable Query Expansion (if needed)
```python
# In code
results = memory_tool.retrieve_memory(query, expand_query=False)

# In retriever
results = retriever.retrieve(query, expand_query=False)
```

### Adjust Query Expansion Threshold
Edit `retriever.py` line 52:
```python
if len(query.split()) > 10:  # Change threshold here
```

### Adjust Number of Results
Edit `tool.py` line 303:
```python
def retrieve_memory(self, query: str, top_k: int = 5, ...):  # Change default
```

## Benefits

1. **Better Precision**: More relevant results for vague queries
2. **Dual Defense**: Both better instructions AND automatic expansion
3. **Transparent**: Logs show when/how queries are expanded
4. **Fallback Safe**: If expansion fails, uses original query
5. **More Results**: 5 results instead of 3 increases chance of finding relevant info

## Testing

### Test Query Expansion
```python
from ymca.tools.memory.retriever import MemoryRetriever

# Test expansion
expanded = retriever.expand_query("authentication")
print(f"Original: 'authentication'")
print(f"Expanded: '{expanded}'")

# Test retrieval with expansion
results = retriever.retrieve("authentication", expand_query=True)
for r in results:
    print(f"- {r['source']}: {r['similarity']:.3f}")
```

### Monitor in Production
Check logs for query expansion:
```
INFO:ymca.tools.memory.retriever:Query expanded: 'authentication' → 'how to configure authentication with examples and troubleshooting'
```

## Limitations

1. **Query expansion adds latency**: ~500ms per query (LLM call)
2. **Model dependency**: Requires model_handler to be available
3. **Context needed**: Works best when model has context about the domain
4. **English-centric**: Optimized for English technical documentation

## Future Improvements

1. **Cache expansions**: Store common query expansions to reduce latency
2. **User feedback loop**: Learn from which expansions led to useful results
3. **Multi-query retrieval**: Generate multiple query variations and merge results
4. **Contextual expansion**: Use conversation history to improve expansions
5. **Hybrid search**: Combine semantic search with keyword matching

