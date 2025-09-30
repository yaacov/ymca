"""Embedding service for the memory module."""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Service for creating and managing embeddings."""

    def __init__(self, model_name: str = "ibm-granite/granite-embedding-english-r2", cache_dir: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """Initialize the embedding service.

        Args:
            model_name: Embedding model name.
            cache_dir: Directory to cache the model files.
            logger: Optional logger instance.
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.logger = logger or logging.getLogger("ymca.memory.embedding")
        self.model: Optional[SentenceTransformer] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the embedding model."""
        try:
            self.logger.info(f"Loading embedding model: {self.model_name}")
            if self.cache_dir:
                self.logger.info(f"Using cache directory: {self.cache_dir}")
                self.model = SentenceTransformer(self.model_name, cache_folder=self.cache_dir)
            else:
                self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Could not load embedding model '{self.model_name}'")

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode text(s) into embeddings."""
        if self.model is None:
            raise RuntimeError("Embedding model is not loaded")

        if isinstance(texts, str):
            texts = [texts]

        try:
            embeddings = self.model.encode(texts, normalize_embeddings=True)
            return embeddings
        except Exception as e:
            self.logger.error(f"Failed to encode texts: {e}")
            raise

    def encode_single(self, text: str) -> List[float]:
        """Encode a single text into embedding."""
        embedding = self.encode(text)
        if len(embedding.shape) == 2:
            return embedding[0].tolist()  # type: ignore
        return embedding.tolist()  # type: ignore

    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))

    def batch_similarity(self, query_embedding: np.ndarray, candidate_embeddings: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between a query and multiple candidates."""
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        candidate_norms = candidate_embeddings / np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
        return np.dot(candidate_norms, query_norm)  # type: ignore

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {"error": "No model loaded"}

        try:
            test_embedding = self.encode("test")
            dimension = test_embedding.shape[-1] if len(test_embedding.shape) > 1 else len(test_embedding)

            return {"model_name": self.model_name, "embedding_dimension": dimension, "max_sequence_length": getattr(self.model, "max_seq_length", "Unknown"), "is_loaded": True}
        except Exception as e:
            return {"model_name": self.model_name, "error": str(e), "is_loaded": False}

    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self.model is not None
