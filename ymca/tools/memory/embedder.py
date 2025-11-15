"""Embedding generation using sentence transformers."""

import logging
from typing import List, Optional
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)


class Embedder:
    """Handles text embedding generation with GPU acceleration support."""
    
    def __init__(
        self, 
        model_name: str = "ibm-granite/granite-embedding-english-r2", 
        cache_folder: str = "models",
        device: Optional[str] = None
    ):
        """
        Initialize embedder.
        
        Args:
            model_name: HuggingFace model name (e.g., "ibm-granite/granite-embedding-english-r2")
                       or local path to model directory
            cache_folder: Directory to cache/store downloaded models (default: "models")
                         Models will be stored in cache_folder/models--org--model-name/
                         Uses sentence-transformers built-in caching
            device: Device to run embeddings on. Options:
                   - None (default): Auto-detect best device (cuda > mps > cpu)
                   - "cuda": NVIDIA GPU
                   - "mps": Apple Silicon GPU (M1/M2/M3)
                   - "cpu": CPU only
        """
        logger.info(f"Loading embedding model: {model_name}...")
        if cache_folder:
            logger.info(f"Using cache folder: {cache_folder}")
        
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        logger.info(f"Using device: {device}")
        
        # Load model with device specification
        self.model = SentenceTransformer(model_name, cache_folder=cache_folder, device=device)
        self.device = device
        
        logger.info("Embedding model loaded")
    
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

