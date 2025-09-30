"""Plan state management and operations."""

import logging
from typing import Any, Dict, List, Optional

from .agent import PlanningAgent
from .models import Plan, PlanStatus
from .persistence import PlanPersistence


class PlanManager:
    """Manages plan state, persistence, and lifecycle operations."""

    def __init__(
        self,
        planning_agent: PlanningAgent,
        plan_persistence: PlanPersistence,
        logger: Optional[logging.Logger] = None,
    ):
        self.planning_agent = planning_agent
        self.plan_persistence = plan_persistence
        self.logger = logger or logging.getLogger("ymca.plan.manager")

        # Current plan tracking
        self.current_plan: Optional[Plan] = None

    async def execute_task(self, user_input: str, system_prompt: Optional[str] = None, context: Optional[List[str]] = None) -> Plan:
        """Execute a task using the planning agent."""
        self.logger.info(f"Executing planning task: {user_input[:100]}...")

        # Create and execute plan
        plan = await self.planning_agent.execute_task(user_input)
        self.current_plan = plan

        # Save the plan
        self.plan_persistence.save_plan(plan)
        self.logger.info(f"💾 Saved plan: {plan.id}")

        return plan

    def get_current_plan_status(self) -> Optional[Dict[str, Any]]:
        """Get the status of the current plan."""
        if not self.current_plan:
            return None

        return {"plan_id": self.current_plan.id, "title": self.current_plan.title, "status": self.current_plan.status.value, "progress": self.current_plan.get_progress_summary(), "created_at": self.current_plan.created_at.isoformat()}

    def list_saved_plans(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List saved plans."""
        return self.plan_persistence.list_plans()[:limit]

    def load_plan(self, plan_id: str) -> Optional[Plan]:
        """Load a saved plan."""
        plan = self.plan_persistence.load_plan(plan_id)
        if plan:
            self.current_plan = plan
            self.logger.info(f"📂 Loaded and set as current plan: {plan_id}")
        return plan

    async def resume_plan(self, plan_id: str) -> str:
        """Resume execution of a saved plan."""
        plan = self.load_plan(plan_id)

        if not plan:
            return f"Plan {plan_id} not found."

        if plan.status == PlanStatus.COMPLETED:
            return f"Plan '{plan.title}' is already completed."

        if plan.status == PlanStatus.FAILED:
            return f"Plan '{plan.title}' has failed. Use retry functionality to retry failed steps."

        self.logger.info(f"Resuming plan: {plan.title}")

        try:
            # Resume execution
            await self.planning_agent.execute_plan(plan)

            # Save updated plan
            self.plan_persistence.save_plan(plan)

            return f"Plan '{plan.title}' resumed successfully."

        except Exception as e:
            self.logger.error(f"Failed to resume plan: {e}")
            return f"Failed to resume plan: {e}"

    def delete_plan(self, plan_id: str) -> bool:
        """Delete a saved plan."""
        success = self.plan_persistence.delete_plan(plan_id)
        if success and self.current_plan and self.current_plan.id == plan_id:
            self.current_plan = None
            self.logger.info(f"Cleared current plan as it was deleted: {plan_id}")
        return success

    def get_persistence_stats(self) -> Dict[str, Any]:
        """Get statistics about stored plans."""
        return self.plan_persistence.get_stats()
