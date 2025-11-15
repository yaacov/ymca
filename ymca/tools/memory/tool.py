"""Main MemoryTool class."""

import logging
import time
from pathlib import Path
from typing import List, Dict

from .chunker import TextChunker
from .embedder import Embedder
from .storage import ChunkStorage
from .vector_store import VectorStore
from .retriever import MemoryRetriever

logger = logging.getLogger(__name__)


class MemoryTool:
    """
    RAG-based memory system with semantic summary retrieval.
    
    Features:
    - Chunking of markdown documents
    - Semantic summary generation for each chunk (declarative statements, not questions)
    - ChromaDB vector store for embeddings
    - File-based chunk storage (one file per chunk)
    - Deduplication of retrieved chunks
    """
    
    # Tool definitions for chat integration
    RETRIEVE_TOOL_DEF = {
        "name": "retrieve_memory",
        "description": (
            "Search documentation and knowledge base for accurate technical information. "
            "This contains official documentation, guides, and examples. "
            "Use this when you need specific details, examples, or documentation that you're not confident about. "
            "For simple or general questions you can answer directly, you don't need to use this tool if you can "
            "answer the question using the chat history."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Detailed technical question to search for in documentation. "
                        "BE SPECIFIC for better results:\n"
                        "✓ GOOD: 'how to configure authentication with step by step examples'\n"
                        "✓ GOOD: 'troubleshooting connection errors with detailed logs'\n"
                        "✓ GOOD: 'API parameters and configuration options reference'\n"
                        "✗ BAD: 'authentication' (too vague)\n"
                        "✗ BAD: 'configuration' (too general)\n"
                        "✗ BAD: 'help' (not specific)\n"
                        "Include: HOW-TO keywords, action verbs (configure/setup/troubleshoot), specific components"
                    )
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 3, range: 1-10)"
                }
            },
            "required": ["query"]
        }
    }
    
    STORE_TOOL_DEF = {
        "name": "store_memory",
        "description": (
            "Store NEW important information in long-term memory. "
            "ONLY use this for information that is NOT already in memory and is worth remembering for future conversations. "
            "DO NOT store trivial facts, temporary information, or duplicates of existing memories. "
            "Always check memory first with retrieve_memory before storing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "information": {
                    "type": "string",
                    "description": "NEW information to store (only if not already in memory)"
                },
                "context": {
                    "type": "string",
                    "description": "Optional context about this information"
                }
            },
            "required": ["information"]
        }
    }
    
    def __init__(
        self, 
        memory_dir: str = "data/tools/memory", 
        model_name: str = "ibm-granite/granite-embedding-english-r2",
        model_cache_dir: str = "models",
        model_handler = None,
        chunk_size: int = 4000,
        overlap: int = 400,
        device: str = None
    ):
        """
        Initialize the memory tool.
        
        Args:
            memory_dir: Directory to store memory data
            model_name: Sentence transformer model name (e.g., "ibm-granite/granite-embedding-english-r2")
                       Default: IBM Granite Embedding English R2 (149M params, 768 dimensions)
            model_cache_dir: Directory to cache embedding models (default: "models")
            model_handler: ModelHandler instance for semantic summary generation (required)
            chunk_size: Chunk size in characters
            overlap: Overlap between chunks in characters
            device: Device for embeddings (None=auto, "cuda", "mps", "cpu")
        """
        if not model_handler:
            raise ValueError("model_handler is required for semantic summary generation")
        
        self.model_handler = model_handler
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
        self.embedder = Embedder(model_name=model_name, cache_folder=model_cache_dir, device=device)
        self.storage = ChunkStorage(self.memory_dir / "chunks")
        self.vector_store = VectorStore(self.memory_dir / "vectors")
        self.retriever = MemoryRetriever(
            storage=self.storage,
            vector_store=self.vector_store,
            embedder=self.embedder,
            model_handler=model_handler
        )
        
        logger.info("✓ Using model handler for semantic summary generation")
    
    def _process_chunk_embedding(self, chunk_text: str, chunk_id: int, max_retries: int = 3) -> tuple:
        """
        Process a chunk: generate semantic summaries and embeddings.
        
        Args:
            chunk_text: Text of the chunk
            chunk_id: Chunk ID
            max_retries: Number of retry attempts
            
        Returns:
            Tuple of (success: bool, summaries: list or None, error: str or None)
        """
        retry_delay = 0.5  # Initial delay in seconds
        
        for attempt in range(max_retries):
            try:
                # Deep clean on retries to ensure fresh state
                deep_clean = (attempt > 0)
                
                # Reset model state before each attempt
                if hasattr(self.retriever, 'model_handler') and self.retriever.model_handler:
                    self.retriever.model_handler.reset_state(deep_clean=deep_clean)
                
                # Brief pause before generation (especially on retries)
                if attempt > 0:
                    time.sleep(retry_delay)
                
                # Generate semantic summaries
                summaries = self.retriever.generate_questions(
                    chunk_text, 
                    num_questions=2
                )
                
                # Generate embeddings for summaries
                summary_embeddings = self.embedder.embed(summaries)
                
                # Store in vector store
                self.vector_store.add_questions(chunk_id, summaries, summary_embeddings)
                
                # Brief pause after successful generation to let model settle
                time.sleep(0.05)
                
                return (True, summaries, None)
                
            except (ValueError, Exception) as e:
                error_msg = str(e)
                if attempt < max_retries - 1:
                    logger.debug(
                        f"Semantic summary generation failed for chunk {chunk_id} "
                        f"(attempt {attempt + 1}/{max_retries}): {error_msg}. "
                        f"Retrying in {retry_delay}s..."
                    )
                    retry_delay *= 2  # Exponential backoff
                else:
                    return (False, None, error_msg)
        
        return (False, None, "Unknown error")
    
    def store_memory(
        self, 
        text: str, 
        source: str = "user_input",
        chunk_size: int = 4000,
        overlap: int = 400,
        retry_interval: int = 5  # Retry pending chunks every N chunks
    ) -> Dict:
        """
        Store a new memory with two-phase commit and retry logic.
        
        Args:
            text: Text to store
            source: Source of the memory
            chunk_size: Chunk size in characters
            overlap: Overlap between chunks
            retry_interval: Retry pending chunks every N chunks
            
        Returns:
            Dictionary with storage statistics
        """
        # Update chunker settings if different
        if chunk_size != self.chunker.chunk_size or overlap != self.chunker.overlap:
            self.chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
        
        # Chunk text
        chunks = self.chunker.chunk_text(text)
        
        # Statistics
        chunks_stored = 0
        questions_generated = 0
        chunks_pending = []
        chunks_failed = []
        chunks_processed = 0
        
        # Phase 1: Store all chunks as "pending" and try to process them
        for i, chunk in enumerate(chunks):
            # Check if chunk already exists
            if self.storage.chunk_exists(chunk):
                continue
            
            # Store chunk with "pending" status
            chunk_meta = self.storage.store_chunk(chunk, source, status="pending")
            chunk_id = chunk_meta['id']
            chunks_stored += 1
            chunks_processed += 1
            
            # Try to process the chunk
            success, questions, error = self._process_chunk_embedding(chunk, chunk_id)
            
            if success:
                # Mark as complete with questions stored in metadata
                self.storage.mark_chunk_complete(chunk_id, questions=questions)
                questions_generated += len(questions)
            else:
                # Keep as pending for retry
                chunks_pending.append((chunk_id, chunk, error))
                logger.debug(f"Chunk {chunk_id} marked as pending: {error}")
            
            # Periodic retry of pending chunks
            if chunks_pending and (i + 1) % retry_interval == 0:
                logger.info(f"Retrying {len(chunks_pending)} pending chunks...")
                chunks_pending = self._retry_pending_chunks(chunks_pending)
                questions_generated += sum(1 for c in self.storage.get_all_chunks() 
                                          if c.get('status') == 'complete') * 2 - questions_generated
        
        # Phase 2: Final retry of all pending chunks
        if chunks_pending:
            logger.info(f"\nFinal retry of {len(chunks_pending)} pending chunks...")
            chunks_pending = self._retry_pending_chunks(chunks_pending, max_retries=3)
        
        # Phase 3: Mark remaining pending chunks as failed
        for chunk_id, chunk_text, error in chunks_pending:
            self.storage.mark_chunk_failed(chunk_id, error=error)
            chunks_failed.append((chunk_id, error))
            logger.warning(f"Chunk {chunk_id} marked as FAILED after all retries: {error}")
        
        # Calculate final statistics
        all_chunks = self.storage.get_all_chunks()
        complete_chunks = [c for c in all_chunks if c.get('status') == 'complete']
        failed_chunks = [c for c in all_chunks if c.get('status') == 'failed']
        
        return {
            "chunks_stored": chunks_stored,
            "chunks_complete": len(complete_chunks),
            "chunks_failed": len(failed_chunks),
            "questions_generated": questions_generated,
            "total_chunks": len(all_chunks),
            "failed_details": chunks_failed
        }
    
    def _retry_pending_chunks(self, pending_chunks: list, max_retries: int = 2) -> list:
        """
        Retry processing pending chunks.
        
        Args:
            pending_chunks: List of (chunk_id, chunk_text, error) tuples
            max_retries: Number of retry attempts per chunk
            
        Returns:
            Updated list of still-pending chunks
        """
        still_pending = []
        
        for chunk_id, chunk_text, prev_error in pending_chunks:
            success, questions, error = self._process_chunk_embedding(
                chunk_text, chunk_id, max_retries=max_retries
            )
            
            if success:
                self.storage.mark_chunk_complete(chunk_id, questions=questions)
                logger.debug(f"✓ Chunk {chunk_id} completed on retry")
            else:
                still_pending.append((chunk_id, chunk_text, error or prev_error))
        
        return still_pending
    
    def retrieve_memory(self, query: str, top_k: int = 3, expand_query: bool = True) -> List[Dict]:
        """
        Retrieve relevant memories.
        
        Args:
            query: Search query
            top_k: Number of results to return (default: 3)
            expand_query: Whether to expand query using LLM (default: True)
            
        Returns:
            List of results with text, source, and similarity
        """
        results = self.retriever.retrieve(query, top_k=top_k, expand_query=expand_query)
        
        if not results:
            logger.info(f"No results found for query: {query}")
            # Return helpful message instead of empty list
            return []
        
        return results
    
    def clear_memory(self):
        """Clear all stored memories."""
        self.storage.clear()
        self.vector_store.clear()
        logger.info("✓ Memory cleared")
    
    def get_stats(self) -> Dict:
        """Get memory statistics."""
        all_chunks = self.storage.get_all_chunks()
        
        return {
            "total_chunks": len(all_chunks),
            "sources": list(set(c['source'] for c in all_chunks)),
            "storage_dir": str(self.memory_dir)
        }
    
    def create_retrieve_tool_function(self):
        """
        Create a tool function for retrieve_memory that can be registered with ChatAPI.
        
        Returns:
            Callable function that formats results for chat
        """
        def retrieve_tool(query: str, max_results: int = 3) -> str:
            """
            Tool function to retrieve memory.
            
            Args:
                query: Search query
                max_results: Maximum number of results (default: 3)
                
            Returns:
                Formatted memory results
            """
            results = self.retrieve_memory(query, top_k=max_results, expand_query=True)
            
            if not results:
                return "No relevant information found in memory"
            
            memory_text = []
            for i, r in enumerate(results, 1):
                memory_text.append(f"{i}. {r['text']} (relevance: {r['similarity']:.2f})")
            
            return "\n".join(memory_text)
        
        return retrieve_tool
    
    def create_store_tool_function(self):
        """
        Create a tool function for store_memory that can be registered with ChatAPI.
        
        Returns:
            Callable function that formats results for chat
        """
        def store_tool(information: str, context: str = "") -> str:
            """
            Tool function to store memory.
            
            Args:
                information: Information to store
                context: Optional context
                
            Returns:
                Success message
            """
            text = information
            if context:
                text = f"{context}: {information}"
            
            result = self.store_memory(text, source="chat")
            chunks_stored = result['chunks_stored']
            
            if chunks_stored == 0:
                return "Information NOT stored - this appears to be a duplicate or already exists in memory. No need to store again."
            elif chunks_stored == 1:
                return "Stored 1 new chunk in memory successfully."
            else:
                return f"Stored {chunks_stored} new chunks in memory successfully."
        
        return store_tool

