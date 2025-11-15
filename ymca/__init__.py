"""
YMCA - Your Memory & Chat Assistant

A framework for building conversational AI with memory and tool capabilities.
"""

__version__ = "0.1.0"

# Expose main modules for convenient imports
from ymca.core.model_handler import ModelHandler
from ymca.tools.memory.tool import MemoryTool
from ymca.chat.api import ChatAPI

__all__ = [
    "ModelHandler",
    "MemoryTool",
    "ChatAPI",
]

