"""Tool registry for managing tools and their execution."""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from ..plan.models import ToolCall, ToolResult


class ToolRegistry:
    """Registry for managing tools and their execution."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.tool_handlers: Dict[str, Callable[..., Any]] = {}
        self.logger = logger or logging.getLogger("ymca.tools.registry")

    def register_tool(self, definition: Dict[str, Any], handler: Callable[..., Any]) -> None:
        """Register a tool with its handler."""
        name = definition["name"]
        self.tools[name] = definition
        self.tool_handlers[name] = handler
        self.logger.info(f"Registered tool: {name}")

    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a tool definition by name."""
        return self.tools.get(name)

    def list_tools(self, category: Optional[str] = None, enabled_only: bool = True) -> List[Dict[str, Any]]:
        """List available tools."""
        tools = list(self.tools.values())

        if enabled_only:
            tools = [tool for tool in tools if tool.get("enabled", True)]

        if category:
            tools = [tool for tool in tools if tool.get("category") == category]

        return tools

    def get_tools_for_llm(self, enabled_only: bool = True) -> List[Dict[str, Any]]:
        """Get tools in OpenAI function calling format."""
        return self.list_tools(enabled_only=enabled_only)

    async def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call and return the result."""
        start_time = time.time()
        call_id = tool_call.get("call_id", "unknown")
        tool_name = tool_call.get("name", "unknown")

        self.logger.info(f"🔧 TOOL REGISTRY: Executing tool '{tool_name}' (call_id: {call_id})")

        # Log parameters with more detail
        parameters = tool_call.get("parameters", {})
        if parameters:
            param_preview = {}
            for k, v in parameters.items():
                if isinstance(v, str) and len(v) > 100:
                    param_preview[k] = f"{v[:100]}... ({len(v)} chars total)"
                else:
                    param_preview[k] = str(v)
            self.logger.info(f"   📝 Parameters: {param_preview}")
        else:
            self.logger.info("   📝 Parameters: None")

        # Check if tool exists
        if tool_name not in self.tools:
            available_tools = list(self.tools.keys())
            error_msg = f"Tool '{tool_name}' not found. Available tools: {available_tools}"
            self.logger.error(f"❌ {error_msg}")
            return {"call_id": call_id, "success": False, "error": error_msg}

        # Get tool info
        tool_def = self.tools[tool_name]
        self.logger.info(f"   🏷️  Tool category: {tool_def.get('category', 'unknown')}")

        # Execute the tool
        try:
            handler = self.tool_handlers[tool_name]

            self.logger.info("   🚀 Starting tool execution...")
            # Handle both sync and async handlers
            if asyncio.iscoroutinefunction(handler):
                self.logger.debug("   🔄 Executing async handler")
                result = await handler(**parameters)
            else:
                self.logger.debug("   🔄 Executing sync handler")
                result = handler(**parameters)

            execution_time = time.time() - start_time

            # Log successful execution with more detail
            if result:
                result_str = str(result)
                result_length = len(result_str)
                result_preview = result_str[:300] if result_length > 300 else result_str
                self.logger.info(f"✅ Tool '{tool_name}' completed in {execution_time:.2f}s")
                self.logger.info(f"   📊 Result length: {result_length} characters")
                self.logger.info(f"   📄 Result preview: {result_preview}{'...' if result_length > 300 else ''}")
                if result_length > 1000:
                    self.logger.debug(f"   📋 Full result: {result}")
            else:
                self.logger.info(f"✅ Tool '{tool_name}' completed in {execution_time:.2f}s")
                self.logger.info("   📄 Result: Empty/None")

            return {"call_id": call_id, "success": True, "result": result, "execution_time": execution_time}

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)
            self.logger.error(f"❌ Tool '{tool_name}' FAILED after {execution_time:.2f}s")
            self.logger.error(f"   💥 Error: {error_msg}")
            self.logger.debug(f"   🔍 Exception details: {repr(e)}")

            return {"call_id": call_id, "success": False, "error": error_msg, "execution_time": execution_time}
