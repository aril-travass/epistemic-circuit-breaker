import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict


class StepLog(TypedDict):
    """Structured turn event log capturing multi-agent execution telemetry."""
    timestamp: float
    iso_time: str
    agent: str
    output: str
    token_count: int
    metadata: Dict[str, Any]


class AgentWorkflowState(TypedDict):
    """
    Shared LangGraph state for the multi-agent reasoning and code generation pipeline.
    Tracks dialogue, generated artifacts, verification steps, and epistemic safety metrics.
    """
    task_prompt: str
    constraints: List[str]
    plan: Optional[str]
    generated_code: Optional[str]
    review_feedback: Optional[str]
    final_code: Optional[str]
    step_logs: List[StepLog]
    fault_injection_enabled: bool
    circuit_breaker_enabled: bool
    drift_history: List[float]
    rollback_count: int


def create_initial_state(
    task_prompt: str,
    constraints: Optional[List[str]] = None,
    fault_injection_enabled: bool = False,
    circuit_breaker_enabled: bool = False,
) -> AgentWorkflowState:
    """Helper to initialize a clean AgentWorkflowState dictionary."""
    return {
        "task_prompt": task_prompt,
        "constraints": constraints or [],
        "plan": None,
        "generated_code": None,
        "review_feedback": None,
        "final_code": None,
        "step_logs": [],
        "fault_injection_enabled": fault_injection_enabled,
        "circuit_breaker_enabled": circuit_breaker_enabled,
        "drift_history": [],
        "rollback_count": 0,
    }


def make_step_log(
    agent: str,
    output: str,
    token_count: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
) -> StepLog:
    """Creates a standardized StepLog record with current timestamp."""
    now = time.time()
    iso = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()
    return {
        "timestamp": now,
        "iso_time": iso,
        "agent": agent,
        "output": output,
        "token_count": token_count,
        "metadata": metadata or {},
    }
