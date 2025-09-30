"""Planning detection and decision making logic."""

import logging
from typing import Optional

from ..config.config import Config


class PlanningDetector:
    """Determines when to use planning based on user input and configuration."""

    def __init__(
        self,
        config: Config,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config
        self.logger = logger or logging.getLogger("ymca.plan.detector")
        self.planning_mode: Optional[bool]

        # Initialize planning mode from configuration
        default_mode = config.get("PLANNING_DEFAULT_MODE", "auto")
        if default_mode == "always":
            self.planning_mode = True
        elif default_mode == "never":
            self.planning_mode = False
        else:  # auto
            self.planning_mode = None  # Use intelligent detection

        self.logger.info(f"Planning detector initialized with mode: {self.get_planning_mode()}")

    def should_use_planning(self, user_input: str) -> bool:
        """Determine if a user request should trigger planning mode."""
        if self.planning_mode is True:
            # Always use planning
            return True
        elif self.planning_mode is False:
            # Never use planning
            return False
        else:
            # Auto mode - use intelligent detection
            return self._detect_planning_need(user_input)

    def _detect_planning_need(self, user_input: str) -> bool:
        """Determine if a user request needs planning based on content analysis."""
        # Keywords that suggest multi-step tasks
        planning_keywords = [
            "plan",
            "steps",
            "analyze",
            "research",
            "find and",
            "search and",
            "create and",
            "build",
            "develop",
            "implement",
            "organize",
            "compare",
            "summarize",
            "investigate",
            "gather information",
            "multi-step",
            "process",
            "workflow",
            "sequence",
            "first",
            "then",
            "after",
            "complex task",
            "detailed analysis",
        ]

        user_input_lower = user_input.lower()

        # Check for planning keywords
        if any(keyword in user_input_lower for keyword in planning_keywords):
            self.logger.info("Planning triggered by keyword detection")
            return True

        # Check for multiple actions (using "and" as indicator)
        if user_input_lower.count(" and ") >= 2:
            self.logger.info("Planning triggered by multiple actions detected")
            return True

        # Check for question length (longer requests often need planning)
        if len(user_input) > 200:
            self.logger.info(f"Planning triggered by long request ({len(user_input)} chars)")
            return True

        self.logger.info("No planning indicators found, using regular chat")
        return False

    def set_planning_mode(self, mode: str) -> None:
        """Set planning mode: 'always', 'never', or 'auto'."""
        if mode == "always":
            self.planning_mode = True
            self.logger.info("Planning mode set to: always use planning")
        elif mode == "never":
            self.planning_mode = False
            self.logger.info("Planning mode set to: never use planning")
        elif mode == "auto":
            self.planning_mode = None
            self.logger.info("Planning mode set to: auto-detect when to use planning")
        else:
            raise ValueError(f"Invalid planning mode: {mode}. Use 'always', 'never', or 'auto'")

    def get_planning_mode(self) -> str:
        """Get current planning mode as string."""
        if self.planning_mode is True:
            return "always"
        elif self.planning_mode is False:
            return "never"
        else:
            return "auto"
