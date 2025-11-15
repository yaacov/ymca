"""File-based storage for text chunks."""

import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ChunkStorage:
    """Stores text chunks as individual files."""
    
    def __init__(self, storage_dir: Path):
        """
        Initialize storage.
        
        Args:
            storage_dir: Directory to store chunks
        """
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata file tracks all chunks
        self.metadata_file = storage_dir / "metadata.json"
        self.metadata = self._load_metadata()
        
        # Validate and repair on startup
        self._validate_and_repair()
    
    def _load_metadata(self) -> Dict:
        """Load metadata from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"chunks": [], "next_id": 0}
    
    def _save_metadata(self):
        """Save metadata to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _get_chunk_hash(self, text: str) -> str:
        """Generate hash for chunk text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _get_chunk_path(self, chunk_id: int) -> Path:
        """Get file path for chunk."""
        return self.storage_dir / f"chunk_{chunk_id:06d}.txt"
    
    def chunk_exists(self, text: str) -> bool:
        """Check if chunk already exists."""
        chunk_hash = self._get_chunk_hash(text)
        return any(c['hash'] == chunk_hash for c in self.metadata['chunks'])
    
    def get_chunk_by_hash(self, chunk_hash: str) -> Optional[Dict]:
        """Get chunk by hash."""
        for chunk in self.metadata['chunks']:
            if chunk['hash'] == chunk_hash:
                return chunk
        return None
    
    def store_chunk(self, text: str, source: str, status: str = "pending") -> Dict:
        """
        Store a text chunk.
        
        Args:
            text: Chunk text
            source: Source identifier
            status: Chunk status - "pending" or "complete"
            
        Returns:
            Chunk metadata dict
        """
        chunk_hash = self._get_chunk_hash(text)
        
        # Check if already exists
        existing = self.get_chunk_by_hash(chunk_hash)
        if existing:
            return existing
        
        # Create new chunk
        chunk_id = self.metadata['next_id']
        self.metadata['next_id'] += 1
        
        chunk_meta = {
            "id": chunk_id,
            "hash": chunk_hash,
            "source": source,
            "length": len(text),
            "status": status
        }
        
        # Save chunk text to file
        chunk_path = self._get_chunk_path(chunk_id)
        with open(chunk_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # Update metadata
        self.metadata['chunks'].append(chunk_meta)
        self._save_metadata()
        
        return chunk_meta
    
    def get_chunk_text(self, chunk_id: int) -> Optional[str]:
        """
        Get chunk text by ID.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            Chunk text or None
        """
        chunk_path = self._get_chunk_path(chunk_id)
        if not chunk_path.exists():
            return None
        
        with open(chunk_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def get_all_chunks(self) -> list:
        """Get all chunk metadata."""
        return self.metadata['chunks']
    
    def get_pending_chunks(self) -> list:
        """Get all chunks with pending status."""
        return [c for c in self.metadata['chunks'] if c.get('status', 'complete') == 'pending']
    
    def mark_chunk_complete(self, chunk_id: int, questions: list = None):
        """
        Mark a chunk as complete.
        
        Args:
            chunk_id: Chunk ID to mark as complete
            questions: Optional list of generated questions to store in metadata
        """
        for chunk in self.metadata['chunks']:
            if chunk['id'] == chunk_id:
                chunk['status'] = 'complete'
                if questions:
                    chunk['questions'] = questions
                self._save_metadata()
                return
    
    def mark_chunk_failed(self, chunk_id: int, error: str = None):
        """
        Mark a chunk as failed.
        
        Args:
            chunk_id: Chunk ID to mark as failed
            error: Optional error message to store in metadata
        """
        for chunk in self.metadata['chunks']:
            if chunk['id'] == chunk_id:
                chunk['status'] = 'failed'
                if error:
                    chunk['error'] = error
                self._save_metadata()
                return
    
    def _validate_and_repair(self):
        """Validate metadata and repair any inconsistencies."""
        if not self.metadata['chunks']:
            return
        
        # Find chunks where metadata exists but file doesn't
        orphaned = []
        for chunk in self.metadata['chunks']:
            chunk_path = self._get_chunk_path(chunk['id'])
            if not chunk_path.exists():
                orphaned.append(chunk['id'])
        
        # Remove orphaned metadata entries
        if orphaned:
            logger.warning(f"Found {len(orphaned)} orphaned chunks (metadata without files), removing...")
            self.metadata['chunks'] = [
                c for c in self.metadata['chunks'] 
                if c['id'] not in orphaned
            ]
            self._save_metadata()
            logger.info(f"Removed {len(orphaned)} orphaned metadata entries")
        
        # Ensure all chunks have a status field (for backwards compatibility)
        updated = False
        for chunk in self.metadata['chunks']:
            if 'status' not in chunk:
                chunk['status'] = 'complete'  # Assume old chunks are complete
                updated = True
        
        if updated:
            self._save_metadata()
            logger.info("Updated chunk metadata with status fields")
    
    def clear(self):
        """Clear all stored chunks."""
        # Remove all chunk files
        for chunk_path in self.storage_dir.glob("chunk_*.txt"):
            chunk_path.unlink()
        
        # Reset metadata
        self.metadata = {"chunks": [], "next_id": 0}
        self._save_metadata()

