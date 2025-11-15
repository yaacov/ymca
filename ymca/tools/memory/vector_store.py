"""ChromaDB vector store for embeddings."""

import logging
import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages embeddings using ChromaDB."""
    
    def __init__(self, storage_dir: Path, collection_name: str = "memory"):
        """
        Initialize vector store.
        
        Args:
            storage_dir: Directory for ChromaDB data
            collection_name: Name of the collection
        """
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(storage_dir / "chroma"),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_questions(
        self, 
        chunk_id: int, 
        questions: List[str], 
        embeddings: np.ndarray
    ):
        """
        Add question embeddings for a chunk.
        
        Args:
            chunk_id: Chunk ID
            questions: List of questions
            embeddings: Question embeddings
        """
        # Create IDs for each question
        ids = [f"chunk_{chunk_id}_q{i}" for i in range(len(questions))]
        
        # Metadata for each question
        metadatas = [
            {"chunk_id": chunk_id, "question": q} 
            for q in questions
        ]
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 3,
        questions_per_chunk: int = 10  # Average is ~9, max 12 (2 questions per 1200-char sub-chunk)
    ) -> List[Dict]:
        """
        Search for similar chunks.
        
        Strategy:
        1. Each chunk has multiple questions (embeddings) - typically 2-8 depending on chunk size
        2. Query may match multiple questions from same chunk
        3. We need to return top_k DISTINCT chunks based on their BEST similarity score
        4. Fetch top_k * questions_per_chunk results to ensure we get enough unique chunks
        
        Args:
            query_embedding: Query embedding
            top_k: Number of unique chunks to return
            questions_per_chunk: Estimated average questions per chunk (default: 6)
            
        Returns:
            List of top_k unique chunks with their best similarity scores, sorted by similarity
        """
        # Check if collection is empty
        collection_size = self.collection.count()
        if collection_size == 0:
            logger.warning("Vector store is empty, no results to return")
            return []
        
        # Calculate how many results to fetch
        # top_k * questions_per_chunk gives us enough to find top_k unique chunks
        fetch_count = min(top_k * questions_per_chunk, collection_size)
        
        # Get top results from vector store
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=fetch_count
        )
        
        # Track best similarity for each unique chunk
        chunk_best_scores = {}  # chunk_id -> similarity
        
        if results['ids'] and len(results['ids']) > 0:
            for i, doc_id in enumerate(results['ids'][0]):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                chunk_id = metadata['chunk_id']
                
                # Convert distance to similarity (1 - distance for cosine)
                similarity = 1 - distance
                
                # Keep only the BEST similarity score for each chunk
                if chunk_id not in chunk_best_scores or similarity > chunk_best_scores[chunk_id]:
                    chunk_best_scores[chunk_id] = similarity
        
        # Convert to list and sort by similarity (best first)
        chunk_results = [
            {"chunk_id": chunk_id, "similarity": similarity}
            for chunk_id, similarity in chunk_best_scores.items()
        ]
        chunk_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Return exactly top_k unique chunks (or fewer if not enough exist)
        return chunk_results[:top_k]
    
    def clear(self):
        """Clear all embeddings."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )

