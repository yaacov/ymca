"""
Tool representation and management for function calling.
"""

from dataclasses import dataclass
from typing import Callable, Dict


@dataclass
class Tool:
    """A tool that the chat API can use."""
    name: str
    description: str
    function: Callable
    parameters: Dict
    
    def to_llm_format(self) -> Dict:
        """Convert to LLM tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

