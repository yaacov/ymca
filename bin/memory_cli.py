#!/usr/bin/env python3
"""
Memory Tool CLI - Command-line interface for memory management.

Commands:
  store     - Store text in memory
  retrieve  - Search and retrieve memories
  clear     - Clear all memories
  stats     - Show memory statistics
  load-docs - Load markdown documents from directory
  list      - List all stored chunks with sources
  
Examples:
  # Store text
  python memory_cli.py store "Python was created by Guido van Rossum"
  
  # Retrieve memories
  python memory_cli.py retrieve "Who created Python?"
  
  # Load documents
  python memory_cli.py load-docs ../kubectl-mtv/docs
  
  # Clear memory
  python memory_cli.py clear --force
  
  # Show stats
  python memory_cli.py stats
  
  # List all chunks
  python memory_cli.py list
  
  # List with preview
  python memory_cli.py list --preview 200
  
  # List chunks from specific file
  python memory_cli.py list --file README.md
  
  # List and group by source
  python memory_cli.py list --group-by-source
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path to import ymca package
sys.path.insert(0, str(Path(__file__).parent.parent))

from ymca.core.model_handler import ModelHandler
from ymca.tools.memory.tool import MemoryTool


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(message)s' if not verbose else '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # Suppress noisy loggers
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('chromadb').setLevel(logging.WARNING)


def cmd_store(args):
    """Store text in memory."""
    print(f"Initializing memory tool (dir: {args.memory_dir})...")
    
    # Initialize
    handler = ModelHandler(model_path=args.model, n_ctx=args.context)
    memory = MemoryTool(
        memory_dir=args.memory_dir,
        model_handler=handler,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        device=args.device
    )
    
    # Store text
    print(f"\nStoring text...")
    result = memory.store_memory(
        text=args.text,
        source=args.source,
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )
    
    print(f"\nStored: {result['chunks_stored']} chunks")
    print(f"  Questions generated: {result['questions_generated']}")
    print(f"  Total chunks in memory: {result['total_chunks']}")


def cmd_retrieve(args):
    """Retrieve memories."""
    print(f"Initializing memory tool (dir: {args.memory_dir})...")
    
    # Initialize
    handler = ModelHandler(model_path=args.model, n_ctx=args.context)
    memory = MemoryTool(
        memory_dir=args.memory_dir,
        model_handler=handler,
        device=args.device
    )
    
    # Retrieve
    print(f"\nSearching for: '{args.query}'")
    results = memory.retrieve_memory(args.query, top_k=args.top_k)
    
    if not results:
        print("\nNo results found")
        return
    
    print(f"\nFound {len(results)} result(s):\n")
    
    for i, result in enumerate(results, 1):
        print(f"{'=' * 70}")
        print(f"Result {i} - Similarity: {result['similarity']:.3f}")
        print(f"Source: {result['source']}")
        print(f"{'=' * 70}")
        print(result['text'])
        print()


def cmd_clear(args):
    """Clear all memories."""
    print(f"Memory directory: {args.memory_dir}")
    
    # Check if exists
    memory_path = Path(args.memory_dir)
    if not memory_path.exists():
        print("Memory directory doesn't exist - nothing to clear")
        return
    
    # Confirm unless --force
    if not args.force:
        response = input("\nWARNING: This will delete all stored memories! Continue? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Cancelled.")
            return
    
    # Initialize and clear
    print("\nInitializing memory tool...")
    handler = ModelHandler(model_path=args.model, n_ctx=args.context)
    memory = MemoryTool(
        memory_dir=args.memory_dir,
        model_handler=handler,
        device=args.device
    )
    
    print("Clearing memory...")
    memory.clear_memory()
    print("\nMemory cleared successfully!")


def cmd_list(args):
    """List all stored chunks."""
    print(f"Memory directory: {args.memory_dir}")
    
    # Check if exists
    memory_path = Path(args.memory_dir)
    if not memory_path.exists():
        print("⚠️  Memory directory doesn't exist")
        return
    
    # Initialize
    print("\nInitializing memory tool...")
    handler = ModelHandler(model_path=args.model, n_ctx=args.context)
    memory = MemoryTool(
        memory_dir=args.memory_dir,
        model_handler=handler
    )
    
    # Get all chunks
    all_chunks = memory.storage.get_all_chunks()
    
    if not all_chunks:
        print("\nNo chunks stored in memory")
        return
    
    # Filter by source if specified
    if args.source:
        all_chunks = [c for c in all_chunks if args.source in c.get('source', '')]
    
    # Filter by file if specified
    if args.file:
        all_chunks = [c for c in all_chunks if args.file in c.get('source', '')]
    
    if not all_chunks:
        print(f"\nNo chunks found matching filters")
        return
    
    print(f"\n{'=' * 70}")
    print(f"STORED CHUNKS ({len(all_chunks)} total)")
    print(f"{'=' * 70}\n")
    
    # Group by source if requested
    if args.group_by_source:
        by_source = {}
        for chunk in all_chunks:
            source = chunk.get('source', 'unknown')
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(chunk)
        
        for source, chunks in sorted(by_source.items()):
            print(f"\n{'=' * 70}")
            print(f"Source: {source} ({len(chunks)} chunks)")
            print(f"{'=' * 70}\n")
            
            for chunk in chunks:
                chunk_id = chunk['id']
                text = memory.storage.get_chunk_text(chunk_id)
                if text:
                    preview = text[:args.preview] if args.preview else text
                    print(f"Chunk {chunk_id}:")
                    print(preview)
                    if args.preview and len(text) > args.preview:
                        print(f"... ({len(text) - args.preview} more characters)")
                    print()
    else:
        # List all chunks
        for chunk in all_chunks:
            chunk_id = chunk['id']
            source = chunk.get('source', 'unknown')
            text = memory.storage.get_chunk_text(chunk_id)
            
            if text:
                print(f"{'=' * 70}")
                print(f"Chunk {chunk_id} - Source: {source}")
                print(f"Length: {chunk.get('length', len(text))} characters")
                print(f"{'=' * 70}")
                
                if args.preview:
                    preview = text[:args.preview]
                    print(preview)
                    if len(text) > args.preview:
                        print(f"\n... ({len(text) - args.preview} more characters)")
                else:
                    print(text)
                print()


def cmd_stats(args):
    """Show memory statistics."""
    print(f"Memory directory: {args.memory_dir}")
    
    # Check if exists
    memory_path = Path(args.memory_dir)
    if not memory_path.exists():
        print("Memory directory doesn't exist")
        return
    
    # Initialize
    print("\nInitializing memory tool...")
    handler = ModelHandler(model_path=args.model, n_ctx=args.context)
    memory = MemoryTool(
        memory_dir=args.memory_dir,
        model_handler=handler
    )
    
    # Get stats
    stats = memory.get_stats()
    
    print(f"\n{'=' * 70}")
    print("MEMORY STATISTICS")
    print(f"{'=' * 70}")
    print(f"\nTotal chunks: {stats['total_chunks']}")
    print(f"Storage directory: {stats['storage_dir']}")
    
    if stats['sources']:
        print(f"\nSources ({len(stats['sources'])}):")
        for source in sorted(stats['sources']):
            print(f"  • {source}")
    
    # Show directory sizes
    chunks_dir = memory_path / "chunks"
    vectors_dir = memory_path / "vectors"
    
    if chunks_dir.exists():
        chunk_files = list(chunks_dir.glob("chunk_*.txt"))
        print(f"\nChunk files: {len(chunk_files)}")
    
    if vectors_dir.exists():
        chroma_dir = vectors_dir / "chroma"
        if chroma_dir.exists():
            print(f"Vector database: ChromaDB")


def cmd_load_docs(args):
    """Load markdown documents."""
    docs_path = Path(args.docs_dir)
    
    if not docs_path.exists():
        print(f"Error: Directory not found: {docs_path}")
        return 1
    
    if not docs_path.is_dir():
        print(f"Error: Not a directory: {docs_path}")
        return 1
    
    print("=" * 70)
    print("LOADING DOCUMENTS INTO MEMORY")
    print("=" * 70)
    print(f"\nDocs directory: {docs_path}")
    print(f"Memory directory: {args.memory_dir}")
    print(f"File pattern: {args.pattern}")
    print(f"Chunk size: {args.chunk_size} characters (storage)")
    print(f"Overlap: {args.overlap} characters")
    print(f"Question generation: ENABLED (2 questions per 1200-char sub-chunk for embeddings)")
    
    # Find files
    md_files = list(docs_path.glob(args.pattern))
    if not md_files:
        print(f"\nError: No files matching '{args.pattern}' found")
        return 1
    
    print(f"\nFound {len(md_files)} file(s)")
    
    # Initialize
    print("\nInitializing...")
    handler = ModelHandler(model_path=args.model, n_ctx=args.context)
    memory = MemoryTool(
        memory_dir=args.memory_dir,
        model_handler=handler,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        device=args.device
    )
    
    # Load documents
    print("\nLoading documents...")
    total_chunks = 0
    total_complete = 0
    total_failed_chunks = 0
    successful = 0
    failed = 0
    all_failed_details = []
    
    for i, md_file in enumerate(md_files, 1):
        print(f"\n[{i}/{len(md_files)}] {md_file.name}")
        
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                print("   Empty file, skipping")
                continue
            
            result = memory.store_memory(
                content,
                source=f"file:{md_file.name}",
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                retry_interval=5
            )
            
            chunks_stored = result['chunks_stored']
            chunks_complete = result.get('chunks_complete', chunks_stored)
            chunks_failed = result.get('chunks_failed', 0)
            
            if chunks_stored > 0:
                status_parts = [f"Stored {chunks_stored} chunks"]
                if chunks_complete < chunks_stored:
                    status_parts.append(f"{chunks_complete} complete")
                if chunks_failed > 0:
                    status_parts.append(f"{chunks_failed} failed")
                
                print(f"   {', '.join(status_parts)}")
                total_chunks += chunks_stored
                total_complete += chunks_complete
                total_failed_chunks += chunks_failed
                successful += 1
                
                # Collect failed details
                if 'failed_details' in result and result['failed_details']:
                    for chunk_id, error in result['failed_details']:
                        all_failed_details.append((md_file.name, chunk_id, error))
            else:
                print(f"   No new chunks (duplicates)")
            
        except Exception as e:
            print(f"   Error: {e}")
            failed += 1
    
    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"\nFiles processed: {len(md_files)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"\nChunks:")
    print(f"  Total stored: {total_chunks}")
    print(f"  Successfully embedded: {total_complete}")
    print(f"  Failed to embed: {total_failed_chunks}")
    
    if total_chunks > 0:
        success_rate = (total_complete / total_chunks) * 100
        print(f"  Success rate: {success_rate:.1f}%")
    
    # Show failed chunk details if any
    if all_failed_details:
        print(f"\n{'=' * 70}")
        print(f"FAILED CHUNKS ({len(all_failed_details)} total)")
        print(f"{'=' * 70}")
        for file_name, chunk_id, error in all_failed_details[:10]:  # Show first 10
            print(f"\n  File: {file_name}")
            print(f"  Chunk ID: {chunk_id}")
            print(f"  Error: {error[:100]}...")
        
        if len(all_failed_details) > 10:
            print(f"\n  ... and {len(all_failed_details) - 10} more failed chunks")
    
    return 0 if (failed == 0 and total_failed_chunks == 0) else 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Memory Tool CLI - Manage memory storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Global options
    parser.add_argument(
        "--memory-dir",
        type=str,
        default="data/tools/memory",
        help="Memory storage directory (default: data/tools/memory)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="models/ibm-granite_granite-4.0-h-tiny/gguf/ibm-granite_granite-4.0-h-tiny-q4_k_m.gguf",
        help="Path to GGUF model"
    )
    
    parser.add_argument(
        "--context",
        type=int,
        default=32768,
        help="Context size in tokens (default: 32768)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Device for embedding model (default: auto-detect - cuda > mps > cpu)"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Store command
    store_parser = subparsers.add_parser('store', help='Store text in memory')
    store_parser.add_argument('text', help='Text to store')
    store_parser.add_argument('--source', default='cli', help='Source identifier (default: cli)')
    store_parser.add_argument('--chunk-size', type=int, default=4000, help='Chunk size for storage (default: 4000)')
    store_parser.add_argument('--overlap', type=int, default=400, help='Chunk overlap (default: 400)')
    
    # Retrieve command
    retrieve_parser = subparsers.add_parser('retrieve', help='Search and retrieve memories')
    retrieve_parser.add_argument('query', help='Search query')
    retrieve_parser.add_argument('--top-k', type=int, default=5, help='Number of results (default: 5)')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear all memories')
    clear_parser.add_argument('--force', action='store_true', help='Skip confirmation')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show memory statistics')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List stored chunks')
    list_parser.add_argument('--source', help='Filter by source (partial match)')
    list_parser.add_argument('--file', help='Filter by filename (partial match)')
    list_parser.add_argument('--preview', type=int, help='Show only first N characters (default: show full text)')
    list_parser.add_argument('--group-by-source', action='store_true', help='Group chunks by source')
    
    # Load-docs command
    load_parser = subparsers.add_parser('load-docs', help='Load markdown documents')
    load_parser.add_argument('docs_dir', help='Directory containing markdown files')
    load_parser.add_argument('--pattern', default='*.md', help='File pattern (default: *.md)')
    load_parser.add_argument('--chunk-size', type=int, default=4000, help='Chunk size for storage (default: 4000)')
    load_parser.add_argument('--overlap', type=int, default=400, help='Chunk overlap (default: 400)')
    
    # Parse args
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    
    # Dispatch command
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'store':
            return cmd_store(args)
        elif args.command == 'retrieve':
            return cmd_retrieve(args)
        elif args.command == 'clear':
            return cmd_clear(args)
        elif args.command == 'stats':
            return cmd_stats(args)
        elif args.command == 'load-docs':
            return cmd_load_docs(args)
        elif args.command == 'list':
            return cmd_list(args)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

