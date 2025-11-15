"""Memory retrieval logic."""

import logging
import time
from typing import List, Dict, Optional
import numpy as np

from .storage import ChunkStorage
from .vector_store import VectorStore
from .embedder import Embedder

logger = logging.getLogger(__name__)


class MemoryRetriever:
    """Handles memory search and retrieval."""
    
    def __init__(
        self, 
        storage: ChunkStorage,
        vector_store: VectorStore,
        embedder: Embedder,
        model_handler=None
    ):
        """
        Initialize retriever.
        
        Args:
            storage: Chunk storage
            vector_store: Vector store
            embedder: Embedder
            model_handler: Optional model handler for semantic summary generation
        """
        self.storage = storage
        self.vector_store = vector_store
        self.embedder = embedder
        self.model_handler = model_handler
    
    def expand_query(self, query: str) -> str:
        """
        Expand a short query into a more comprehensive search query using LLM.
        
        Args:
            query: Original search query
            
        Returns:
            Expanded query string
        """
        if not self.model_handler:
            return query
        
        # Only expand if query is short (likely too general)
        if len(query.split()) > 10:
            logger.debug(f"Query already detailed ({len(query.split())} words), skipping expansion")
            return query
        
        try:
            # Reset model state
            self.model_handler.reset_state()
            
            prompt = f"""Expand this search query to be more specific for documentation search.
Keep it concise (15-25 words). Add HOW-TO words and context, but do NOT invent specific tools, commands, or implementation details not mentioned in the original query.

Examples:
- "authentication" → "how to configure authentication setup and credentials"
- "create provider" → "how to create and configure a new provider"
- "troubleshooting errors" → "how to troubleshoot and diagnose error messages"

Original query: {query}

Expanded query:"""
            
            # Generate expanded query
            expanded = self.model_handler.llm.create_completion(
                prompt,
                max_tokens=50,
                temperature=0.1,  # Lower temperature to reduce invention
                stop=["\n", "?"],
                echo=False
            )
            
            expanded_text = expanded['choices'][0]['text'].strip()
            
            # Clean up and validate
            if expanded_text and len(expanded_text) > len(query):
                logger.info(f"Query expanded: '{query}' → '{expanded_text}'")
                return expanded_text
            else:
                logger.debug("Query expansion didn't improve query, using original")
                return query
                
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}, using original query")
            return query
    
    def retrieve(self, query: str, top_k: int = 3, expand_query: bool = False) -> List[Dict]:
        """
        Retrieve relevant memories.
        
        Args:
            query: Search query
            top_k: Number of results
            expand_query: Whether to expand the query using LLM (default: False)
            
        Returns:
            List of results with text, source, and similarity (deduplicated by chunk_id)
        """
        # Optionally expand query for better matching
        search_query = query
        if expand_query and self.model_handler:
            search_query = self.expand_query(query)
        
        # Embed query
        query_embedding = self.embedder.embed_single(search_query)
        
        # Search vector store (should already deduplicate, but we'll verify)
        results = self.vector_store.search(query_embedding, top_k=top_k)
        
        # Load chunk texts with additional deduplication check
        retrieved = []
        seen_chunk_ids = set()
        
        for result in results:
            chunk_id = result['chunk_id']
            
            # Skip if we've already seen this chunk_id
            if chunk_id in seen_chunk_ids:
                logger.warning(f"Duplicate chunk_id {chunk_id} in vector store results, skipping")
                continue
            
            seen_chunk_ids.add(chunk_id)
            text = self.storage.get_chunk_text(chunk_id)
            
            if text:
                # Get chunk metadata
                chunk_meta = next(
                    (c for c in self.storage.get_all_chunks() if c['id'] == chunk_id),
                    {}
                )
                
                retrieved.append({
                    "text": text,
                    "source": chunk_meta.get('source', 'unknown'),
                    "similarity": result['similarity'],
                    "chunk_id": chunk_id
                })
        
        return retrieved
    
    def generate_questions(self, chunk: str, num_questions: int = 2, max_chunk_size: int = 1200) -> List[str]:
        """
        Generate semantic summaries for a chunk using LLM.
        
        Creates declarative statements that capture key information for better retrieval.
        If the chunk is larger than max_chunk_size, it will be split into sub-chunks
        and num_questions semantic summaries will be generated for EACH sub-chunk.
        
        Args:
            chunk: Text chunk
            num_questions: Number of semantic summaries per (sub)chunk
            max_chunk_size: Maximum size for summary generation (default: 1200 chars)
                           This is chosen to match embedding model capacity:
                           - IBM Granite Embedding English R2: 512 tokens max
                           - 1200 chars ≈ 300 tokens (safe margin)
            
        Returns:
            List of semantic summaries (declarative statements)
            
        Raises:
            ValueError: If model_handler is not provided
            Exception: If summary generation fails
        """
        if not self.model_handler:
            raise ValueError("model_handler is required for semantic summary generation")
        
        # If chunk is small enough, generate summaries directly
        if len(chunk) <= max_chunk_size:
            # Reset model state before generating to avoid decode errors
            self.model_handler.reset_state()
            return self.model_handler.generate_questions(chunk, num_questions=num_questions)
        
        # Split large chunk into sub-chunks
        sub_chunks = []
        start = 0
        overlap = 200
        
        while start < len(chunk):
            end = min(start + max_chunk_size, len(chunk))
            
            # Try to break at sentence boundary
            if end < len(chunk):
                for separator in ['. ', '.\n', '! ', '?\n', '\n\n']:
                    last_sep = chunk[start:end].rfind(separator)
                    if last_sep > max_chunk_size // 2:
                        end = start + last_sep + len(separator)
                        break
            
            sub_chunk = chunk[start:end].strip()
            if sub_chunk:
                sub_chunks.append(sub_chunk)
            
            start = end - overlap if end < len(chunk) else len(chunk)
        
        logger.debug(f"Split {len(chunk)} char chunk into {len(sub_chunks)} sub-chunks")
        
        # Generate semantic summaries for EACH sub-chunk
        all_summaries = []
        for i, sub_chunk in enumerate(sub_chunks):
            max_retries = 2
            success = False
            
            for retry in range(max_retries):
                try:
                    # Reset model state before each sub-chunk to avoid decode errors
                    # Use deep clean on retries
                    self.model_handler.reset_state(deep_clean=(retry > 0))
                    
                    # Add brief pause between sub-chunks and retries
                    if i > 0 or retry > 0:
                        time.sleep(0.2 if retry == 0 else 0.5)
                    
                    summaries = self.model_handler.generate_questions(sub_chunk, num_questions=num_questions)
                    all_summaries.extend(summaries)
                    success = True
                    break
                    
                except Exception as e:
                    error_msg = str(e)
                    logger.warning(
                        f"Semantic summary generation failed for sub-chunk {i+1}/{len(sub_chunks)} "
                        f"(attempt {retry+1}/{max_retries}): {error_msg}"
                    )
                    
                    # If it's a decode error and not last retry, do aggressive cleanup
                    if ("llama_decode" in error_msg or "returned -1" in error_msg) and retry < max_retries - 1:
                        logger.info("Performing deep cleanup before retry...")
                        try:
                            self.model_handler.reset_state(deep_clean=True)
                            time.sleep(0.5)
                        except Exception as reset_error:
                            logger.error(f"Failed to reset model state: {reset_error}")
            
            if not success:
                logger.error(
                    f"Failed to generate summaries for sub-chunk {i+1}/{len(sub_chunks)} "
                    f"after {max_retries} attempts. Continuing with remaining sub-chunks."
                )
        
        if not all_summaries:
            raise ValueError(
                f"Failed to generate any summaries from {len(sub_chunks)} sub-chunks. "
                f"This may indicate the chunk is too large or complex."
            )
        
        logger.debug(f"Generated {len(all_summaries)} summaries from {len(sub_chunks)} sub-chunks")
        return all_summaries

