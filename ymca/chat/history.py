"""
Conversation history management with sliding window.
"""

from typing import List, Dict, Optional
from .message import Message


class ConversationHistory:
    """Manages conversation history with a sliding window."""
    
    def __init__(self, max_messages: int = 20):
        """
        Initialize conversation history.
        
        Args:
            max_messages: Maximum number of messages to keep (default: 20)
        """
        self.max_messages = max_messages
        self.messages: List[Message] = []
        self.system_message: Optional[Message] = None
    
    def set_system_message(self, content: str):
        """Set the system message."""
        self.system_message = Message(role="system", content=content)
    
    def add_message(self, message: Message):
        """Add a message to history."""
        self.messages.append(message)
        
        # Keep only last N messages (excluding system message)
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def add_user_message(self, content: str):
        """Add a user message."""
        self.add_message(Message(role="user", content=content))
    
    def add_assistant_message(self, content: str, tool_calls: Optional[List[Dict]] = None):
        """Add an assistant message."""
        self.add_message(Message(role="assistant", content=content, tool_calls=tool_calls))
    
    def add_tool_message(self, name: str, content: str, tool_call_id: str):
        """Add a tool response message."""
        self.add_message(Message(role="tool", content=content, name=name, tool_call_id=tool_call_id))
    
    def get_messages(self) -> List[Dict]:
        """Get messages in LLM format."""
        messages = []
        if self.system_message:
            messages.append(self.system_message.to_dict())
        messages.extend([m.to_dict() for m in self.messages])
        return messages
    
    def clear(self):
        """Clear conversation history (keeps system message)."""
        self.messages = []
    
    def get_summary(self) -> str:
        """Get a summary of the conversation."""
        if not self.messages:
            return "No messages yet"
        
        summary = []
        for msg in self.messages[-5:]:  # Last 5 messages
            role = msg.role.upper()
            content = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
            summary.append(f"[{role}]: {content}")
        
        return "\n".join(summary)

