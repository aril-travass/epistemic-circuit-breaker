import os
import time
from collections import deque
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()


class CircuitState(str, Enum):
    CLOSED = "CLOSED"        # Normal execution, monitor epistemic confidence
    OPEN = "OPEN"            # Breaker tripped due to hallucination/uncertainty; execution rolled back/halted
    HALF_OPEN = "HALF_OPEN"  # Trial probe state; validating recovery attempts


class StepEvaluation(BaseModel):
    step_id: str
    confidence: float
    entropy: Optional[float] = None
    passed: bool
    state_before: CircuitState
    state_after: CircuitState
    reason: str
    timestamp: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Checkpoint(BaseModel):
    checkpoint_id: str
    step_name: str
    state_snapshot: Dict[str, Any]
    confidence: float
    timestamp: float = Field(default_factory=time.time)


class EpistemicCircuitBreaker:
    """
    Epistemic Circuit Breaker for Autonomous AI Agents.
    Monitors epistemic confidence, hallucination entropy, and factual grounding
    across agent reasoning steps. Automatically trips to OPEN and triggers state
    rollback when epistemic divergence exceeds safety thresholds.
    """

    def __init__(
        self,
        threshold: Optional[float] = None,
        sliding_window_size: Optional[int] = None,
        recovery_patience: Optional[int] = None,
    ):
        self.threshold: float = float(
            threshold
            if threshold is not None
            else os.getenv("EPISTEMIC_THRESHOLD", "0.75")
        )
        self.sliding_window_size: int = int(
            sliding_window_size
            if sliding_window_size is not None
            else os.getenv("SLIDING_WINDOW_SIZE", "5")
        )
        self.recovery_patience: int = int(
            recovery_patience
            if recovery_patience is not None
            else os.getenv("RECOVERY_PATIENCE", "3")
        )

        self.state: CircuitState = CircuitState.CLOSED
        self.failure_count: int = 0
        self.consecutive_successes: int = 0
        self.trip_reason: Optional[str] = None
        self.history: deque[StepEvaluation] = deque(maxlen=self.sliding_window_size)
        self.checkpoints: List[Checkpoint] = []
        self.last_rollback_checkpoint: Optional[Checkpoint] = None

    def save_checkpoint(
        self, step_name: str, state_snapshot: Dict[str, Any], confidence: float
    ) -> Checkpoint:
        """Saves a grounded safe state snapshot to enable rollback upon failure."""
        checkpoint = Checkpoint(
            checkpoint_id=f"chk_{int(time.time() * 1000)}_{len(self.checkpoints)}",
            step_name=step_name,
            state_snapshot=state_snapshot,
            confidence=confidence,
        )
        self.checkpoints.append(checkpoint)
        return checkpoint

    def get_last_safe_checkpoint(self) -> Optional[Checkpoint]:
        """Returns the most recent checkpoint with verified confidence >= threshold."""
        for cp in reversed(self.checkpoints):
            if cp.confidence >= self.threshold:
                return cp
        return self.checkpoints[-1] if self.checkpoints else None

    def trip(self, reason: str) -> None:
        """Forcibly trip the circuit breaker into OPEN state and mark rollback."""
        self.state = CircuitState.OPEN
        self.trip_reason = reason
        self.consecutive_successes = 0
        self.last_rollback_checkpoint = self.get_last_safe_checkpoint()

    def reset(self) -> None:
        """Resets the circuit breaker to CLOSED normal operating status."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.consecutive_successes = 0
        self.trip_reason = None
        self.last_rollback_checkpoint = None

    def transition_to_half_open(self, reason: str = "Initiating recovery trial") -> None:
        """Transition from OPEN to HALF_OPEN to test candidate fixes."""
        self.state = CircuitState.HALF_OPEN
        self.consecutive_successes = 0
        self.trip_reason = reason

    def rollback(self) -> Optional[Dict[str, Any]]:
        """Rolls back agent state to the last verified safe checkpoint."""
        safe_cp = self.get_last_safe_checkpoint()
        self.last_rollback_checkpoint = safe_cp
        if safe_cp:
            return {
                "checkpoint_id": safe_cp.checkpoint_id,
                "step_name": safe_cp.step_name,
                "state_snapshot": safe_cp.state_snapshot,
                "confidence": safe_cp.confidence,
                "timestamp": safe_cp.timestamp,
            }
        return None

    def evaluate_step(
        self,
        step_id: str,
        content: str = "",
        confidence: float = 1.0,
        entropy: Optional[float] = None,
        state_snapshot: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates epistemic metrics for an agent step.
        Trips breaker and initiates rollback if confidence < threshold or entropy spike detected.
        """
        metadata = metadata or {}
        state_before = self.state
        passed = confidence >= self.threshold

        # If uncertainty entropy is provided and high, cross-check
        if entropy is not None and entropy > (1.0 - self.threshold + 0.1):
            passed = False
            reason = f"Entropy spike detected ({entropy:.3f} > {1.0 - self.threshold + 0.1:.3f})"
        elif not passed:
            reason = f"Epistemic confidence ({confidence:.3f}) fell below threshold ({self.threshold:.3f})"
        else:
            reason = f"Grounded step passed confidence check ({confidence:.3f} >= {self.threshold:.3f})"

        rolled_back = False
        rollback_data: Optional[Dict[str, Any]] = None

        if self.state == CircuitState.CLOSED:
            if passed:
                self.failure_count = 0
                if state_snapshot is not None:
                    self.save_checkpoint(step_id, state_snapshot, confidence)
            else:
                self.failure_count += 1
                self.trip(reason=reason)
                rollback_data = self.rollback()
                rolled_back = True

        elif self.state == CircuitState.HALF_OPEN:
            if passed:
                self.consecutive_successes += 1
                if self.consecutive_successes >= self.recovery_patience:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.trip_reason = None
                    reason += f" - Breaker recovered to CLOSED after {self.consecutive_successes} consecutive successes."
                if state_snapshot is not None:
                    self.save_checkpoint(step_id, state_snapshot, confidence)
            else:
                self.trip(reason=f"Recovery failed in HALF_OPEN: {reason}")
                rollback_data = self.rollback()
                rolled_back = True

        elif self.state == CircuitState.OPEN:
            # When already OPEN, reject or signal active trip
            reason = f"Circuit breaker is OPEN ({self.trip_reason}). Step rejected."
            rollback_data = self.rollback()
            rolled_back = True

        state_after = self.state

        eval_record = StepEvaluation(
            step_id=step_id,
            confidence=confidence,
            entropy=entropy,
            passed=passed,
            state_before=state_before,
            state_after=state_after,
            reason=reason,
            metadata=metadata,
        )
        self.history.append(eval_record)

        return {
            "step_id": step_id,
            "passed": passed,
            "confidence": confidence,
            "entropy": entropy,
            "threshold": self.threshold,
            "state_before": state_before.value,
            "state_after": state_after.value,
            "reason": reason,
            "rolled_back": rolled_back,
            "rollback_target": rollback_data,
            "failure_count": self.failure_count,
            "consecutive_successes": self.consecutive_successes,
        }

    def get_status(self) -> Dict[str, Any]:
        """Telemetry view of circuit breaker status."""
        safe_cp = self.get_last_safe_checkpoint()
        return {
            "state": self.state.value,
            "is_tripped": self.state == CircuitState.OPEN,
            "threshold": self.threshold,
            "sliding_window_size": self.sliding_window_size,
            "recovery_patience": self.recovery_patience,
            "failure_count": self.failure_count,
            "consecutive_successes": self.consecutive_successes,
            "trip_reason": self.trip_reason,
            "history_count": len(self.history),
            "sliding_window_history": [item.model_dump() for item in self.history],
            "total_checkpoints": len(self.checkpoints),
            "last_safe_checkpoint": safe_cp.model_dump() if safe_cp else None,
            "last_rollback_checkpoint": (
                self.last_rollback_checkpoint.model_dump()
                if self.last_rollback_checkpoint
                else None
            ),
        }
