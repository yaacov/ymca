"""
Message representation for chat conversations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict


@dataclass
class Message:
    """A single message in the conversation."""
    role: str  # 'system', 'user', 'assistant', 'tool'
    content: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None  # For tool responses
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for LLM API."""
        msg = {"role": self.role}
        if self.content:
            msg["content"] = self.content
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.name:
            msg["name"] = self.name
        return msg

