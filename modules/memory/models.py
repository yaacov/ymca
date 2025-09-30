"""Data models for the memory module."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class MemoryQuestion:
    """Represents a question that a memory can answer."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    embedding: Optional[List[float]] = None
    memory_id: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "text": self.text,
            "embedding": self.embedding,
            "memory_id": self.memory_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryQuestion":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            text=data.get("text", ""),
            embedding=data.get("embedding"),
            memory_id=data.get("memory_id", ""),
        )


@dataclass
class Memory:
    """Represents a stored memory with its associated questions."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    summary: str = ""
    questions: List[MemoryQuestion] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    url: Optional[str] = None
    line_numbers: Optional[str] = None  # e.g., "10-15", "10", "10,12-15"

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "content": self.content,
            "summary": self.summary,
            "questions": [q.to_dict() for q in self.questions],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": self.tags,
            "url": self.url,
            "line_numbers": self.line_numbers,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Memory":
        """Create from dictionary."""
        questions = [MemoryQuestion.from_dict(q) for q in data.get("questions", [])]

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            content=data.get("content", ""),
            summary=data.get("summary", ""),
            questions=questions,
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat())),
            tags=data.get("tags", []),
            url=data.get("url"),
            line_numbers=data.get("line_numbers"),
        )


@dataclass
class MemorySearchResult:
    """Represents a memory search result with relevance score."""

    memory: Memory
    relevance_score: float = 0.0
    matched_questions: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "memory": self.memory.to_dict(),
            "relevance_score": self.relevance_score,
            "matched_questions": self.matched_questions,
        }
