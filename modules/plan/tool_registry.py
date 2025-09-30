"""Backward compatibility wrapper for tool registry and creation functions."""

# Import everything from the new tools module for backward compatibility
from ..tools.filesystem import create_filesystem_tools
from ..tools.memory import create_memory_tools
from ..tools.registry import ToolRegistry
from ..tools.synthesis import create_synthesis_tools
from ..tools.web import create_web_tools

# Re-export everything to maintain API compatibility
__all__ = ["ToolRegistry", "create_web_tools", "create_filesystem_tools", "create_memory_tools", "create_synthesis_tools"]
