"""Enhanced chat manager with planning capabilities."""

import logging
from typing import Any, Dict, List, Optional

from ..config.config import Config
from ..filesystem.filesystem_manager import FilesystemManager
from ..llm.llm import LLM
from ..memory.memory_manager import MemoryManager
from ..plan.agent import PlanningAgent
from ..plan.models import Plan
from ..plan.persistence import PlanPersistence
from ..plan.plan_manager import PlanManager
from ..plan.planning_detector import PlanningDetector
from ..plan.response_processor import PlanResponseProcessor
from ..plan.tool_setup import PlanningToolSetup
from ..tools.registry import ToolRegistry
from ..web.web_browser import WebBrowser
from .chat_manager import ChatManager


class PlanningChatManager(ChatManager):
    """Chat manager with integrated multi-step planning capabilities."""

    def __init__(
        self,
        llm: LLM,
        config: Config,
        web_browser: Optional[WebBrowser] = None,
        memory_manager: Optional[MemoryManager] = None,
        filesystem_manager: Optional[FilesystemManager] = None,
        history_window: int = 5,
        max_history_tokens: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(llm, history_window, max_history_tokens, logger)

        self.config = config

        # Initialize planning components
        self.tool_registry = ToolRegistry(logger=self.logger)
        self.planning_agent = PlanningAgent(llm, self.tool_registry, logger=self.logger)
        self.plan_persistence = PlanPersistence(persistence_dir=config.get("PLANNING_PERSISTENCE_DIR", "./data/plans"), logger=self.logger)

        # Initialize extracted components
        self.planning_detector = PlanningDetector(config, logger=self.logger)
        self.response_processor = PlanResponseProcessor(llm, logger=self.logger)
        self.plan_manager = PlanManager(self.planning_agent, self.plan_persistence, logger=self.logger)

        # Initialize tool setup and configure all tools
        self.tool_setup = PlanningToolSetup(
            config=config,
            tool_registry=self.tool_registry,
            llm=llm,
            web_browser=web_browser,
            memory_manager=memory_manager,
            filesystem_manager=filesystem_manager,
            logger=self.logger,
        )
        self.tool_setup.setup_all_tools()

        self.logger.info("Planning chat manager initialized")

    def set_planning_mode(self, mode: str) -> None:
        """Set planning mode: 'always', 'never', or 'auto'."""
        self.planning_detector.set_planning_mode(mode)

    def get_planning_mode(self) -> str:
        """Get current planning mode as string."""
        return self.planning_detector.get_planning_mode()

    async def send_message_with_planning(self, user_input: str, system_prompt: Optional[str] = None, context: Optional[List[str]] = None) -> str:
        """Send a message that may trigger planning based on configuration and task complexity."""

        # Determine if this requires planning using the detector
        should_plan = self.planning_detector.should_use_planning(user_input)

        if should_plan:
            return await self._handle_planning_request(user_input, system_prompt, context)
        else:
            # Use regular chat
            return self.send_message(user_input, system_prompt, context)

    async def force_planning(self, user_input: str, system_prompt: Optional[str] = None, context: Optional[List[str]] = None) -> str:
        """Force planning regardless of current mode settings."""
        self.logger.info("Forcing planning execution regardless of mode")
        return await self._handle_planning_request(user_input, system_prompt, context)

    async def _handle_planning_request(self, user_input: str, system_prompt: Optional[str] = None, context: Optional[List[str]] = None) -> str:
        """Handle a request that requires planning."""

        self.logger.info(f"Handling planning request: {user_input[:100]}...")

        try:
            # Execute the task using the plan manager
            plan = await self.plan_manager.execute_task(user_input, system_prompt, context)

            # Generate response based on plan execution
            display_response = self.response_processor.generate_plan_response(plan)

            # Extract clean final answer for chat history (without planning metadata)
            clean_answer = self.response_processor.extract_clean_answer(plan)

            # Log what gets displayed vs stored
            self.logger.info(f"📺 Display response length: {len(display_response)} chars")
            self.logger.info(f"📚 Chat history answer length: {len(clean_answer)} chars")
            self.logger.debug(f"📺 Display response preview: {display_response[:200]}...")
            self.logger.debug(f"📚 Chat history preview: {clean_answer[:200]}...")

            # Store clean answer in conversation history
            self._store_conversation(user_input, clean_answer)

            return display_response

        except Exception as e:
            self.logger.error(f"Planning execution failed: {e}")
            error_response = f"I encountered an error while planning and executing your request: {e}\n\n"
            error_response += "Let me try to help you with a simpler approach."

            # Fall back to regular chat
            fallback_response = self.send_message(user_input, system_prompt, context)
            return error_response + fallback_response

    def get_current_plan_status(self) -> Optional[Dict[str, Any]]:
        """Get the status of the current plan."""
        return self.plan_manager.get_current_plan_status()

    def list_saved_plans(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List saved plans."""
        return self.plan_manager.list_saved_plans(limit)

    def load_plan(self, plan_id: str) -> Optional[Plan]:
        """Load a saved plan."""
        return self.plan_manager.load_plan(plan_id)

    async def resume_plan(self, plan_id: str) -> str:
        """Resume execution of a saved plan."""
        result = await self.plan_manager.resume_plan(plan_id)

        # If resume was successful and returned a plan result, generate proper response
        if not result.startswith("Plan") and not result.startswith("Failed"):
            # This is a success case, generate proper display response
            if self.plan_manager.current_plan:
                return self.response_processor.generate_plan_response(self.plan_manager.current_plan)

        return result

    def delete_plan(self, plan_id: str) -> bool:
        """Delete a saved plan."""
        return self.plan_manager.delete_plan(plan_id)

    def get_available_tools(self) -> list[str]:
        """Get list of available tools."""
        return self.tool_setup.get_available_tools()

    def get_tool_stats(self) -> dict[str, Any]:
        """Get statistics about registered tools."""
        return self.tool_setup.get_tool_stats()

    def get_persistence_stats(self) -> dict[str, Any]:
        """Get statistics about stored plans."""
        return self.plan_manager.get_persistence_stats()
