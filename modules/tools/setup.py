"""Tool setup and configuration for various operations."""

import logging
from typing import Any, Optional

from ..config.config import Config
from ..filesystem.filesystem_manager import FilesystemManager
from ..llm.llm import LLM
from ..memory.memory_manager import MemoryManager
from ..web.web_browser import WebBrowser
from .filesystem import create_filesystem_tools
from .memory import create_memory_tools
from .registry import ToolRegistry
from .synthesis import create_synthesis_tools
from .web import create_web_tools


class ToolSetup:
    """Handles setup and configuration of tools for various operations."""

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
        self.config = config
        self.tool_registry = tool_registry
        self.llm = llm
        self.web_browser = web_browser
        self.memory_manager = memory_manager
        self.filesystem_manager = filesystem_manager
        self.logger = logger or logging.getLogger("ymca.tools.setup")

    def setup_all_tools(self) -> None:
        """Set up all available tools based on configuration."""
        self.logger.info("🔧 Setting up tools...")

        self._setup_web_tools()
        self._setup_filesystem_tools()
        self._setup_memory_tools()
        self._setup_synthesis_tools()

        # Log total tool count
        total_tools = len(self.tool_registry.tools)
        self.logger.info(f"🛠️  TOTAL tools available: {total_tools}")

        # List all available tools for debugging
        all_tool_names = list(self.tool_registry.tools.keys())
        self.logger.debug(f"🔍 All available tools: {all_tool_names}")

    def _setup_web_tools(self) -> None:
        """Set up web browser tools."""
        web_enabled = self.config.get("ENABLE_WEB_TOOLS", True)
        if web_enabled and self.web_browser:
            self.logger.info("🌐 Setting up web tools...")
            web_tools = create_web_tools(self.web_browser)
            web_tool_names = []
            for tool_def, handler in web_tools:
                self.tool_registry.register_tool(tool_def, handler)
                web_tool_names.append(tool_def["name"])
            self.logger.info(f"🌐 Registered {len(web_tools)} web tools: {web_tool_names}")
        elif web_enabled and not self.web_browser:
            self.logger.warning("🌐 Web tools enabled but no web_browser provided!")
        else:
            self.logger.info("🌐 Web tools disabled by configuration")

    def _setup_filesystem_tools(self) -> None:
        """Set up filesystem tools."""
        fs_enabled = self.config.get("ENABLE_FILESYSTEM_TOOLS", True)
        if fs_enabled and self.filesystem_manager:
            self.logger.info("📁 Setting up filesystem tools...")
            fs_tools = create_filesystem_tools(self.filesystem_manager)
            fs_tool_names = []
            for tool_def, handler in fs_tools:
                self.tool_registry.register_tool(tool_def, handler)
                fs_tool_names.append(tool_def["name"])
            self.logger.info(f"📁 Registered {len(fs_tools)} filesystem tools: {fs_tool_names}")
        elif fs_enabled and not self.filesystem_manager:
            self.logger.warning("📁 Filesystem tools enabled but no filesystem_manager provided!")
        else:
            self.logger.info("📁 Filesystem tools disabled by configuration")

    def _setup_memory_tools(self) -> None:
        """Set up memory management tools."""
        memory_enabled = self.config.get("ENABLE_MEMORY_TOOLS", True)
        if memory_enabled and self.memory_manager:
            self.logger.info("🧠 Setting up memory tools...")
            memory_tools = create_memory_tools(self.memory_manager)
            memory_tool_names = []
            for tool_def, handler in memory_tools:
                self.tool_registry.register_tool(tool_def, handler)
                memory_tool_names.append(tool_def["name"])
            self.logger.info(f"🧠 Registered {len(memory_tools)} memory tools: {memory_tool_names}")
        elif memory_enabled and not self.memory_manager:
            self.logger.warning("🧠 Memory tools enabled but no memory_manager provided!")
        else:
            self.logger.info("🧠 Memory tools disabled by configuration")

    def _setup_synthesis_tools(self) -> None:
        """Set up synthesis tools."""
        self.logger.info("✨ Setting up synthesis tools...")
        synthesis_tools = create_synthesis_tools(self.llm)
        synthesis_tool_names = []
        for tool_def, handler in synthesis_tools:
            self.tool_registry.register_tool(tool_def, handler)
            synthesis_tool_names.append(tool_def["name"])
        self.logger.info(f"✨ Registered {len(synthesis_tools)} synthesis tools: {synthesis_tool_names}")

    def get_available_tools(self) -> list[str]:
        """Get list of available tools."""
        tools = self.tool_registry.list_tools()
        return [f"{tool['name']} ({tool['category']})" for tool in tools]

    def get_tool_stats(self) -> dict[str, Any]:
        """Get statistics about registered tools."""
        tools = self.tool_registry.list_tools(enabled_only=False)

        stats: dict[str, Any] = {"total_tools": len(tools), "enabled_tools": len([t for t in tools if t.get("enabled", True)]), "by_category": {}}

        for tool in tools:
            category = tool.get("category", "general")
            if category not in stats["by_category"]:
                stats["by_category"][category] = {"total": 0, "enabled": 0}

            stats["by_category"][category]["total"] += 1
            if tool.get("enabled", True):
                stats["by_category"][category]["enabled"] += 1

        return stats
