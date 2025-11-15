#!/usr/bin/env python3
"""
Test script for query expansion improvements.

This demonstrates the before/after behavior of query expansion.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ymca.core.model_handler import ModelHandler
from ymca.tools.memory.tool import MemoryTool

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

def test_query_expansion():
    """Test query expansion with various queries."""
    
    print("=" * 80)
    print("Query Expansion Test")
    print("=" * 80)
    
    # Initialize model handler
    print("\n1. Initializing model handler...")
    model_handler = ModelHandler(
        model_path="models/ibm-granite_granite-4.0-h-tiny/granite-4.0-h-tiny-q8_0.gguf"
    )
    
    # Initialize memory tool
    print("2. Initializing memory tool...")
    memory_tool = MemoryTool(
        memory_dir="data/tools/memory",
        model_handler=model_handler
    )
    
    # Test queries (generic examples that work with any technical docs)
    test_queries = [
        "authentication",
        "configuration",
        "troubleshooting errors",
        "API reference",
        "installation guide",
        "best practices",
    ]
    
    print("\n3. Testing query expansion:")
    print("-" * 80)
    
    for query in test_queries:
        print(f"\n📝 Original Query: '{query}'")
        
        # Show expansion
        expanded = memory_tool.retriever.expand_query(query)
        print(f"🔍 Expanded Query: '{expanded}'")
        
        # Show what gets retrieved
        print(f"\n   Retrieving with expansion...")
        results = memory_tool.retrieve_memory(query, top_k=3, expand_query=True)
        
        if results:
            print(f"   ✓ Found {len(results)} results:")
            for i, r in enumerate(results, 1):
                source_file = r['source'].replace('file:', '')
                print(f"     {i}. {source_file} (similarity: {r['similarity']:.3f})")
                # Show first 100 chars of text
                text_preview = r['text'][:100].replace('\n', ' ')
                print(f"        Preview: {text_preview}...")
        else:
            print(f"   ✗ No results found")
        
        print()
    
    # Compare with vs without expansion
    print("\n4. Comparing expansion ON vs OFF:")
    print("-" * 80)
    
    test_query = "authentication"
    
    print(f"\n📝 Query: '{test_query}'")
    
    print("\n   A. WITH expansion:")
    results_with = memory_tool.retrieve_memory(test_query, top_k=3, expand_query=True)
    if results_with:
        for i, r in enumerate(results_with, 1):
            source = r['source'].replace('file:', '')
            print(f"      {i}. {source} ({r['similarity']:.3f})")
    
    print("\n   B. WITHOUT expansion:")
    results_without = memory_tool.retrieve_memory(test_query, top_k=3, expand_query=False)
    if results_without:
        for i, r in enumerate(results_without, 1):
            source = r['source'].replace('file:', '')
            print(f"      {i}. {source} ({r['similarity']:.3f})")
    
    # Show difference
    sources_with = {r['source'] for r in results_with}
    sources_without = {r['source'] for r in results_without}
    
    if sources_with != sources_without:
        print("\n   💡 Results differ! Expansion improved matching.")
        only_with_expansion = sources_with - sources_without
        if only_with_expansion:
            print(f"      New sources found: {only_with_expansion}")
    else:
        print("\n   ℹ️  Results are the same (query was already good)")
    
    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)

if __name__ == "__main__":
    test_query_expansion()

