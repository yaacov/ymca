"""Backward compatibility wrapper for planning tool setup."""

import logging
from typing import Any, Optional

from ..config.config import Config
from ..filesystem.filesystem_manager import FilesystemManager
from ..llm.llm import LLM
from ..memory.memory_manager import MemoryManager
from ..tools.registry import ToolRegistry
from ..tools.setup import ToolSetup
from ..web.web_browser import WebBrowser


class PlanningToolSetup:
    """Backward compatibility wrapper for planning tool setup.

    This class maintains the existing API while delegating to the new ToolSetup class.
    """

    def __init__(
        self,
        config: Config,
        tool_registry: ToolRegistry,
        llm: LLM,
        web_browser: Optional[WebBrowser] = None,
        memory_manager: Optional[MemoryManager] = None,
        filesystem_manager: Optional[FilesystemManager] = None,
        logger: Optional[logging.Logger] = None,
    ):
        # Create a new config dict that maps old planning-specific keys to new generic keys
        planning_config = self._create_planning_config(config)

        # Create the new ToolSetup instance
        self._tool_setup = ToolSetup(
            config=planning_config,
            tool_registry=tool_registry,
            llm=llm,
            web_browser=web_browser,
            memory_manager=memory_manager,
            filesystem_manager=filesystem_manager,
            logger=logger or logging.getLogger("ymca.plan.tool_setup"),
        )

    def _create_planning_config(self, original_config: Config) -> Config:
        """Create a config that maps old planning keys to new generic keys."""

        # Create a simple config wrapper that maps old keys to new ones
        class ConfigMapper(Config):
            def __init__(self, original: Config):
                self._original = original

            def get(self, key: str, default: Any = None) -> Any:
                # Map old planning-specific keys to new generic keys
                mapping = {
                    "ENABLE_WEB_TOOLS": "PLANNING_ENABLE_WEB_TOOLS",
                    "ENABLE_FILESYSTEM_TOOLS": "PLANNING_ENABLE_FILESYSTEM_TOOLS",
                    "ENABLE_MEMORY_TOOLS": "PLANNING_ENABLE_MEMORY_TOOLS",
                }

                # If this is a new generic key, check if we have the old planning version
                old_key = mapping.get(key, key)
                return self._original.get(old_key, default)

        return ConfigMapper(original_config)

    def setup_all_tools(self) -> None:
        """Set up all available tools based on configuration."""
        return self._tool_setup.setup_all_tools()

    def get_available_tools(self) -> list[str]:
        """Get list of available tools."""
        return self._tool_setup.get_available_tools()

    def get_tool_stats(self) -> dict[str, Any]:
        """Get statistics about registered tools."""
        return self._tool_setup.get_tool_stats()
