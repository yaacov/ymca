"""Embedding generation using sentence transformers."""

import logging
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """Handles text embedding generation."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedder.
        
        Args:
            model_name: Name of sentence transformer model
        """
        logger.info(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        logger.info("✓ Embedding model loaded")
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Numpy array of embeddings
        """
        return self.model.encode(texts, show_progress_bar=False)
    
    def embed_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array embedding
        """
        return self.model.encode([text], show_progress_bar=False)[0]

