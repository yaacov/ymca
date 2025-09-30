"""Simplified data models for the plan module."""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class PlanStatus(Enum):
    """Status of a plan."""

    CREATED = "created"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


@dataclass
class Goal:
    """Represents the objective we're trying to achieve."""

    description: str
    success_criteria: List[str] = field(default_factory=list)
    achieved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "success_criteria": self.success_criteria,
            "achieved": self.achieved,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Goal":
        return cls(
            description=data["description"],
            success_criteria=data.get("success_criteria", []),
            achieved=data.get("achieved", False),
        )


@dataclass
class ToolFailure:
    """Track tool failures to enable smart retry logic."""
    tool_name: str
    failure_count: int = 0
    last_error: str = ""
    first_failure_time: Optional[datetime] = None
    last_failure_time: Optional[datetime] = None

    def record_failure(self, error: str) -> None:
        """Record a new failure."""
        self.failure_count += 1
        self.last_error = error
        now = datetime.now()
        if self.first_failure_time is None:
            self.first_failure_time = now
        self.last_failure_time = now

    def should_retry(self, max_retries: int = 3) -> bool:
        """Check if tool should be retried."""
        return self.failure_count < max_retries

    def is_repeatedly_failing(self, threshold: int = 2) -> bool:
        """Check if tool is repeatedly failing."""
        return self.failure_count >= threshold


@dataclass
class PlanExecution:
    """Detailed execution information for a single iteration."""
    iteration: int
    timestamp: datetime
    reasoning_prompt: str
    reasoning_response: str
    reasoning_result: Dict[str, Any]
    action: str
    action_parameters: Dict[str, Any]
    action_result: Dict[str, Any]
    observation: str
    success: bool
    error: Optional[str] = None
    strategy_suggestion: Optional[str] = None  # For guidance on next steps


@dataclass
class Plan:
    """A simplified plan with goal and execution tracking."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    goal: Optional[Goal] = None
    status: PlanStatus = PlanStatus.CREATED
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: str = ""
    error: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    # Evolving answer that builds up with each step
    evolving_answer: str = ""
    knowledge_pieces: List[str] = field(default_factory=list)
    # Detailed execution tracking for debugging
    executions: List[PlanExecution] = field(default_factory=list)
    max_iterations: int = 10
    current_iteration: int = 0
    # Tool failure tracking for smart strategy selection
    tool_failures: Dict[str, ToolFailure] = field(default_factory=dict)
    # URL tracking to prevent duplicate web page reads
    read_urls: Set[str] = field(default_factory=set)

    def is_url_already_read(self, url: str) -> bool:
        """Check if a URL has already been read."""
        return url in self.read_urls
    
    def mark_url_as_read(self, url: str) -> None:
        """Mark a URL as read."""
        self.read_urls.add(url)

    def is_completed(self) -> bool:
        """Check if plan is completed."""
        return self.status == PlanStatus.COMPLETED

    def has_failed_steps(self) -> bool:
        """Check if plan has failed."""
        return self.status == PlanStatus.FAILED

    def record_tool_failure(self, tool_name: str, error: str) -> None:
        """Record a tool failure for strategic planning."""
        if tool_name not in self.tool_failures:
            self.tool_failures[tool_name] = ToolFailure(tool_name=tool_name)
        self.tool_failures[tool_name].record_failure(error)

    def should_retry_tool(self, tool_name: str, max_retries: int = 3) -> bool:
        """Check if a tool should be retried based on failure history."""
        if tool_name not in self.tool_failures:
            return True  # No failures yet, safe to try
        return self.tool_failures[tool_name].should_retry(max_retries)

    def get_repeatedly_failing_tools(self) -> List[str]:
        """Get list of tools that are repeatedly failing."""
        return [
            name for name, failure in self.tool_failures.items()
            if failure.is_repeatedly_failing()
        ]

    def get_strategy_guidance(self, available_tools: List[str]) -> Optional[str]:
        """Get strategic guidance based on tool failure patterns."""
        failing_tools = self.get_repeatedly_failing_tools()
        if not failing_tools:
            return None

        # Build guidance based on which tools are failing
        guidance_parts = [f"⚠️  Strategic Guidance: The following tools have repeatedly failed: {', '.join(failing_tools)}"]
        
        # Suggest alternatives based on failing tools
        alternatives = []
        if "search_web" in failing_tools and "read_webpage" in available_tools:
            alternatives.append("Try reading specific documentation URLs with 'read_webpage'")
        if "read_webpage" in failing_tools and "search_web" in available_tools:
            alternatives.append("Try a different search approach or search terms")
        if len(failing_tools) >= 2 and "synthesize_information" in available_tools:
            alternatives.append("Consider using 'synthesize_information' with any accumulated knowledge")

        if alternatives:
            guidance_parts.append(f"Suggested alternatives: {'; '.join(alternatives)}")
        else:
            guidance_parts.append("Consider providing a final answer based on any information already gathered")

        return "\n".join(guidance_parts)

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get a summary of plan progress."""
        return {
            "total_steps": 1,  # Simplified - single execution step
            "completed": 1 if self.status == PlanStatus.COMPLETED else 0,
            "failed": 1 if self.status == PlanStatus.FAILED else 0,
            "pending": 1 if self.status == PlanStatus.CREATED else 0,
            "in_progress": 1 if self.status == PlanStatus.IN_PROGRESS else 0,
            "progress_percentage": 100 if self.status == PlanStatus.COMPLETED else 0,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "goal": self.goal.to_dict() if self.goal else None,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
            "context": self.context,
            "evolving_answer": self.evolving_answer,
            "knowledge_pieces": self.knowledge_pieces,
            "executions": [self._execution_to_dict(exec) for exec in self.executions],
            "max_iterations": self.max_iterations,
            "current_iteration": self.current_iteration,
            "tool_failures": self._tool_failures_to_dict(),
        }

    def _execution_to_dict(self, execution: PlanExecution) -> Dict[str, Any]:
        """Convert PlanExecution to dictionary."""
        return {
            "iteration": execution.iteration,
            "timestamp": execution.timestamp.isoformat(),
            "reasoning_prompt": execution.reasoning_prompt,
            "reasoning_response": execution.reasoning_response,
            "reasoning_result": execution.reasoning_result,
            "action": execution.action,
            "action_parameters": execution.action_parameters,
            "action_result": execution.action_result,
            "observation": execution.observation,
            "success": execution.success,
            "error": execution.error,
            "strategy_suggestion": execution.strategy_suggestion,
        }

    def _tool_failures_to_dict(self) -> Dict[str, Any]:
        """Convert tool failures to dictionary."""
        return {
            name: {
                "tool_name": failure.tool_name,
                "failure_count": failure.failure_count,
                "last_error": failure.last_error,
                "first_failure_time": failure.first_failure_time.isoformat() if failure.first_failure_time else None,
                "last_failure_time": failure.last_failure_time.isoformat() if failure.last_failure_time else None,
            }
            for name, failure in self.tool_failures.items()
        }

    def _tool_failures_from_dict(self, data: Dict[str, Any]) -> Dict[str, ToolFailure]:
        """Create tool failures from dictionary."""
        failures = {}
        for name, failure_data in data.items():
            failure = ToolFailure(
                tool_name=failure_data["tool_name"],
                failure_count=failure_data["failure_count"],
                last_error=failure_data["last_error"],
            )
            if failure_data.get("first_failure_time"):
                failure.first_failure_time = datetime.fromisoformat(failure_data["first_failure_time"])
            if failure_data.get("last_failure_time"):
                failure.last_failure_time = datetime.fromisoformat(failure_data["last_failure_time"])
            failures[name] = failure
        return failures

    def _execution_from_dict(self, data: Dict[str, Any]) -> PlanExecution:
        """Create PlanExecution from dictionary."""
        return PlanExecution(
            iteration=data["iteration"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            reasoning_prompt=data["reasoning_prompt"],
            reasoning_response=data["reasoning_response"],
            reasoning_result=data["reasoning_result"],
            action=data["action"],
            action_parameters=data["action_parameters"],
            action_result=data["action_result"],
            observation=data["observation"],
            success=data["success"],
            error=data.get("error"),
            strategy_suggestion=data.get("strategy_suggestion"),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Plan":
        """Create from dictionary."""
        plan = cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            goal=Goal.from_dict(data["goal"]) if data.get("goal") else None,
            status=PlanStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            result=data.get("result", ""),
            error=data.get("error"),
            context=data.get("context", {}),
            evolving_answer=data.get("evolving_answer", ""),
            knowledge_pieces=data.get("knowledge_pieces", []),
            max_iterations=data.get("max_iterations", 10),
            current_iteration=data.get("current_iteration", 0),
        )

        # Handle executions
        if "executions" in data:
            plan.executions = [plan._execution_from_dict(exec_data) for exec_data in data["executions"]]

        # Handle tool failures
        if "tool_failures" in data:
            plan.tool_failures = plan._tool_failures_from_dict(data["tool_failures"])

        if data.get("started_at"):
            plan.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            plan.completed_at = datetime.fromisoformat(data["completed_at"])

        return plan

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "Plan":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


# Compatibility with existing planning module
StepStatus = PlanStatus  # For compatibility
ToolCall = Dict[str, Any]  # Simplified
ToolResult = Dict[str, Any]  # Simplified
PlanStep = Dict[str, Any]  # Simplified
ToolDefinition = Dict[str, Any]  # Simplified
