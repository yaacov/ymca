"""Data models for web browser functionality."""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SearchResult:
    """Represents a web search result."""

    title: str
    url: str
    description: str
    relevance_score: Optional[float] = None


@dataclass
class WebPage:
    """Represents extracted web page content."""

    url: str
    title: str
    content: str
    links: List[str]
    metadata: Dict[str, Any]
    extraction_timestamp: float

    @classmethod
    def create_now(cls, url: str, title: str, content: str, links: List[str], metadata: Dict[str, Any]) -> "WebPage":
        """Create a WebPage with current timestamp."""
        return cls(url=url, title=title, content=content, links=links, metadata=metadata, extraction_timestamp=time.time())


@dataclass
class ContentQuality:
    """Represents content quality assessment results."""

    score: float  # 0.0 to 1.0
    reasons: List[str]
    length: int

    @property
    def is_high_quality(self) -> bool:
        """Check if content is considered high quality."""
        return self.score > 0.8

    @property
    def is_low_quality(self) -> bool:
        """Check if content is considered low quality."""
        return self.score < 0.3


@dataclass
class ContentCharacteristics:
    """Represents analyzed content characteristics."""

    is_github: bool
    is_documentation: bool
    is_technical: bool
    has_code: bool

    @property
    def extraction_strategy(self) -> str:
        """Determine the best extraction strategy based on characteristics."""
        if self.is_github:
            return "github"
        elif self.is_documentation:
            return "documentation"
        else:
            return "general"
