"""Simplified plan persistence for the plan module."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import Plan


class PlanPersistence:
    """Simplified plan persistence manager."""

    def __init__(self, persistence_dir: str = "./data/plans", logger: Optional[logging.Logger] = None):
        self.persistence_dir = Path(persistence_dir)
        self.logger = logger or logging.getLogger("ymca.plan.persistence")

        # Create persistence directory if it doesn't exist
        self.persistence_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Plan persistence initialized: {self.persistence_dir}")

    def save_plan(self, plan: Plan) -> bool:
        """Save a plan to disk."""
        try:
            plan_file = self.persistence_dir / f"{plan.id}.json"
            with open(plan_file, "w", encoding="utf-8") as f:
                json.dump(plan.to_dict(), f, indent=2)

            self.logger.info(f"💾 Saved plan: {plan.id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save plan {plan.id}: {e}")
            return False

    def load_plan(self, plan_id: str) -> Optional[Plan]:
        """Load a plan from disk."""
        try:
            plan_file = self.persistence_dir / f"{plan_id}.json"
            if not plan_file.exists():
                self.logger.warning(f"Plan file not found: {plan_id}")
                return None

            with open(plan_file, "r", encoding="utf-8") as f:
                plan_data = json.load(f)

            plan = Plan.from_dict(plan_data)
            self.logger.info(f"📂 Loaded plan: {plan_id}")
            return plan

        except Exception as e:
            self.logger.error(f"Failed to load plan {plan_id}: {e}")
            return None

    def list_plans(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List saved plans."""
        try:
            plans = []
            for plan_file in self.persistence_dir.glob("*.json"):
                try:
                    with open(plan_file, "r", encoding="utf-8") as f:
                        plan_data = json.load(f)

                    # Return summary info
                    plans.append(
                        {
                            "id": plan_data["id"],
                            "title": plan_data.get("title", ""),
                            "description": plan_data.get("description", "")[:100] + "..." if len(plan_data.get("description", "")) > 100 else plan_data.get("description", ""),
                            "status": plan_data.get("status", "unknown"),
                            "created_at": plan_data.get("created_at"),
                            "completed_at": plan_data.get("completed_at"),
                        }
                    )

                except Exception as e:
                    self.logger.warning(f"Failed to read plan file {plan_file}: {e}")
                    continue

            # Sort by creation date (newest first)
            plans.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            return plans[:limit]

        except Exception as e:
            self.logger.error(f"Failed to list plans: {e}")
            return []

    def delete_plan(self, plan_id: str) -> bool:
        """Delete a plan from disk."""
        try:
            plan_file = self.persistence_dir / f"{plan_id}.json"
            if plan_file.exists():
                plan_file.unlink()
                self.logger.info(f"🗑️  Deleted plan: {plan_id}")
                return True
            else:
                self.logger.warning(f"Plan file not found for deletion: {plan_id}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to delete plan {plan_id}: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored plans."""
        try:
            plan_files = list(self.persistence_dir.glob("*.json"))
            total_plans = len(plan_files)

            status_counts: Dict[str, int] = {}
            for plan_file in plan_files:
                try:
                    with open(plan_file, "r", encoding="utf-8") as f:
                        plan_data = json.load(f)
                    status = plan_data.get("status", "unknown")
                    status_counts[status] = status_counts.get(status, 0) + 1
                except Exception:
                    continue

            return {
                "total_plans": total_plans,
                "status_breakdown": status_counts,
                "storage_path": str(self.persistence_dir),
            }

        except Exception as e:
            self.logger.error(f"Failed to get persistence stats: {e}")
            return {"error": str(e)}
