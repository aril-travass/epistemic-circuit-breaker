import copy
import logging
import os
import time
from collections import deque
from enum import Enum
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Ensure environment variables are loaded
load_dotenv()

logger = logging.getLogger("epistemic-circuit-breaker.circuit_breaker")

# Robust imports supporting execution from root or backend directory
try:
    from backend.middleware.claim_extractor import (
        AtomicClaim,
        ClaimExtractionResult,
        ClaimType,
        extract_atomic_claims,
    )
    from backend.middleware.drift_scorer import (
        ClaimEvaluation,
        DriftEvaluationResult,
        EpistemicDriftScorer,
        get_drift_scorer,
    )
except ImportError:
    try:
        from middleware.claim_extractor import (
            AtomicClaim,
            ClaimExtractionResult,
            ClaimType,
            extract_atomic_claims,
        )
        from middleware.drift_scorer import (
            ClaimEvaluation,
            DriftEvaluationResult,
            EpistemicDriftScorer,
            get_drift_scorer,
        )
    except ImportError:
        from .claim_extractor import (
            AtomicClaim,
            ClaimExtractionResult,
            ClaimType,
            extract_atomic_claims,
        )
        from .drift_scorer import (
            ClaimEvaluation,
            DriftEvaluationResult,
            EpistemicDriftScorer,
            get_drift_scorer,
        )


class CircuitState(str, Enum):
    CLOSED = "CLOSED"        # Normal execution, monitor epistemic confidence / drift
    OPEN = "OPEN"            # Breaker tripped due to hallucination/uncertainty; execution rolled back/halted
    HALF_OPEN = "HALF_OPEN"  # Trial probe state; validating recovery attempts


class Checkpoint(BaseModel):
    checkpoint_id: str
    step_name: str
    state_snapshot: Dict[str, Any]
    drift_score: float = 0.0
    confidence: float = 1.0
    timestamp: float = Field(default_factory=time.time)
    active_claims: List[Dict[str, Any]] = Field(default_factory=list)


class CircuitBreakerEvaluation(BaseModel):
    action: str  # "VERIFIED" or "TRIP_AND_ROLLBACK"
    step_name: str
    drift_score: float
    threshold: float
    tripped: bool
    circuit_state: str
    extracted_claims: List[Dict[str, Any]] = Field(default_factory=list)
    drift_result: Optional[Dict[str, Any]] = None
    corrective_directive: Optional[str] = None
    rollback_target: Optional[Dict[str, Any]] = None
    purged_output: Optional[str] = None
    reason: str = ""
    timestamp: float = Field(default_factory=time.time)


class StepEvaluation(BaseModel):
    """Telemetry structure for individual step evaluation records."""
    step_id: str
    confidence: float = 1.0
    entropy: Optional[float] = None
    passed: bool
    state_before: CircuitState
    state_after: CircuitState
    reason: str
    timestamp: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EpistemicCircuitBreaker:
    """
    Epistemic Circuit Breaker for Autonomous AI Agents.
    Monitors epistemic drift, semantic uncertainty, and factual grounding across agent steps.
    Extracts atomic claims via claim_extractor and evaluates drift via EpistemicDriftScorer.
    Automatically trips to OPEN and triggers state rollback when epistemic drift D_t >= threshold (tau).
    """

    def __init__(
        self,
        threshold: Optional[float] = None,
        alpha: Optional[float] = None,
        drift_scorer: Optional[EpistemicDriftScorer] = None,
        sliding_window_size: Optional[int] = None,
        recovery_patience: Optional[int] = None,
    ):
        self.threshold: float = float(
            threshold
            if threshold is not None
            else os.getenv("EPISTEMIC_THRESHOLD", "0.50")
        )
        self.alpha: float = float(
            alpha
            if alpha is not None
            else os.getenv("DRIFT_ALPHA", "0.70")
        )
        self.sliding_window_size: int = int(
            sliding_window_size
            if sliding_window_size is not None
            else os.getenv("SLIDING_WINDOW_SIZE", "5")
        )
        self.recovery_patience: int = int(
            recovery_patience
            if recovery_patience is not None
            else os.getenv("RECOVERY_PATIENCE", "2")
        )

        self._drift_scorer = drift_scorer
        self.state: CircuitState = CircuitState.CLOSED
        self.failure_count: int = 0
        self.consecutive_successes: int = 0
        self.trip_reason: Optional[str] = None
        self.history: deque[StepEvaluation] = deque(maxlen=self.sliding_window_size)
        self.checkpoints: List[Checkpoint] = []
        self.last_rollback_checkpoint: Optional[Checkpoint] = None
        self.total_trips: int = 0
        self.total_rollbacks: int = 0

    @property
    def drift_scorer(self) -> EpistemicDriftScorer:
        """Lazily initialize or access EpistemicDriftScorer."""
        if self._drift_scorer is None:
            self._drift_scorer = get_drift_scorer()
            if self.threshold is not None:
                self._drift_scorer.threshold = self.threshold
            if self.alpha is not None:
                self._drift_scorer.alpha = self.alpha
        return self._drift_scorer

    def save_checkpoint(
        self,
        step_name: str,
        state_snapshot: Dict[str, Any],
        drift_score: float = 0.0,
        active_claims: Optional[List[Dict[str, Any]]] = None,
    ) -> Checkpoint:
        """Saves a grounded safe state snapshot to enable rollback upon failure."""
        # Sanitize state snapshot to avoid recursive or deep mutation
        safe_state = {}
        for k, v in state_snapshot.items():
            if k in ("safe_checkpoints", "step_logs"):
                # Shallow copy container to prevent infinite nesting
                safe_state[k] = list(v) if isinstance(v, list) else v
            else:
                try:
                    safe_state[k] = copy.deepcopy(v)
                except Exception:
                    safe_state[k] = v

        checkpoint = Checkpoint(
            checkpoint_id=f"chk_{int(time.time() * 1000)}_{len(self.checkpoints)}",
            step_name=step_name,
            state_snapshot=safe_state,
            drift_score=drift_score,
            confidence=max(0.0, 1.0 - drift_score),
            timestamp=time.time(),
            active_claims=active_claims or [],
        )
        self.checkpoints.append(checkpoint)
        logger.info(
            "Safe checkpoint saved: id=%s step=%s drift=%.4f (total: %d)",
            checkpoint.checkpoint_id,
            step_name,
            drift_score,
            len(self.checkpoints),
        )
        return checkpoint

    def get_last_safe_checkpoint(self) -> Optional[Checkpoint]:
        """Returns the most recent checkpoint with verified drift_score < threshold."""
        for cp in reversed(self.checkpoints):
            if cp.drift_score < self.threshold:
                return cp
        return self.checkpoints[-1] if self.checkpoints else None

    def trip(self, reason: str) -> None:
        """Forcibly trip the circuit breaker into OPEN state and record telemetry."""
        self.state = CircuitState.OPEN
        self.trip_reason = reason
        self.total_trips += 1
        self.failure_count += 1
        self.consecutive_successes = 0
        self.last_rollback_checkpoint = self.get_last_safe_checkpoint()
        logger.warning("Circuit breaker TRIPPED to OPEN: %s", reason)

    def reset(self) -> None:
        """Resets the circuit breaker to CLOSED normal operating status."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.consecutive_successes = 0
        self.trip_reason = None
        self.last_rollback_checkpoint = None
        logger.info("Circuit breaker reset to CLOSED.")

    def transition_to_half_open(self, reason: str = "Initiating recovery trial") -> None:
        """Transition from OPEN to HALF_OPEN to test candidate fixes."""
        self.state = CircuitState.HALF_OPEN
        self.consecutive_successes = 0
        self.trip_reason = reason
        logger.info("Circuit breaker transitioned to HALF_OPEN: %s", reason)

    def rollback(self) -> Optional[Dict[str, Any]]:
        """Rolls back agent state to the last verified safe checkpoint."""
        safe_cp = self.get_last_safe_checkpoint()
        self.last_rollback_checkpoint = safe_cp
        if safe_cp:
            self.total_rollbacks += 1
            logger.info("Rolling back to checkpoint %s (step: %s)", safe_cp.checkpoint_id, safe_cp.step_name)
            return safe_cp.model_dump(mode="json")
        logger.warning("Rollback requested but no safe checkpoint exists.")
        return None

    def evaluate_transition(
        self,
        state: Dict[str, Any],
        step_name: str,
        agent_output: str,
        threshold: Optional[float] = None,
    ) -> CircuitBreakerEvaluation:
        """
        Main LangGraph arbitration transition evaluator.
        1. Extracts atomic claims from agent_output.
        2. Computes drift score D_t using EpistemicDriftScorer.
        3. If D_t >= tau: trips breaker, rolls back state, generates corrective directive.
        4. If D_t < tau: verifies step, registers safe checkpoint.
        """
        active_threshold = threshold if threshold is not None else self.threshold
        task_prompt = state.get("task_prompt", "")
        raw_constraints = state.get("constraints", [])
        corrective_constraints = state.get("corrective_constraints", [])
        combined_constraints = list(raw_constraints) + list(corrective_constraints)

        # 1. Extract atomic claims
        claims_result: ClaimExtractionResult = extract_atomic_claims(
            text=agent_output,
            context_prompt=task_prompt
        )
        claims = claims_result.claims
        claims_dump = [c.model_dump(mode="json") for c in claims]

        # Collect verified checkpoint statements for premise anchoring
        checkpoint_claim_statements: List[str] = []
        for cp in self.checkpoints:
            for ac in cp.active_claims:
                stmt = ac.get("statement") if isinstance(ac, dict) else getattr(ac, "statement", None)
                if stmt and stmt not in checkpoint_claim_statements:
                    checkpoint_claim_statements.append(stmt)

        # 2. Evaluate Epistemic Drift D_t
        drift_res: DriftEvaluationResult = self.drift_scorer.evaluate_claims(
            claims=claims,
            task_prompt=task_prompt,
            constraints=combined_constraints,
            verified_checkpoint_claims=checkpoint_claim_statements,
            threshold=active_threshold,
        )
        drift_score = drift_res.drift_score
        drift_res_dump = drift_res.model_dump(mode="json")

        # Step record for legacy telemetry
        state_before = self.state
        passed = not drift_res.tripped

        if drift_res.tripped:
            # TRIP & ROLLBACK
            reason = (
                f"Epistemic drift D_t={drift_score:.4f} exceeded threshold tau={active_threshold:.4f} "
                f"at step '{step_name}'."
            )
            self.trip(reason=reason)
            rollback_data = self.rollback()

            corrective_directive = drift_res.corrective_directive
            if not corrective_directive:
                if drift_res.critical_claim:
                    corrective_directive = (
                        f"Refrain from unverified assumption: '{drift_res.critical_claim.statement}'. "
                        "Adhere strictly to verified specifications and Python standard libraries."
                    )
                else:
                    corrective_directive = (
                        f"Epistemic drift detected ({drift_score:.4f} >= {active_threshold:.4f}). "
                        "Eliminate unverified imports and ensure grounded implementations."
                    )

            purged_output = (
                f"[CIRCUIT BREAKER TRIPPED - OUTPUT PURGED DUE TO EPISTEMIC DRIFT {drift_score:.4f} >= {active_threshold:.4f}]\n"
                f"Corrective Directive: {corrective_directive}"
            )

            eval_record = StepEvaluation(
                step_id=step_name,
                confidence=max(0.0, 1.0 - drift_score),
                entropy=drift_res.semantic_entropy,
                passed=False,
                state_before=state_before,
                state_after=self.state,
                reason=reason,
                metadata={"drift_score": drift_score, "claims_count": len(claims)},
            )
            self.history.append(eval_record)

            return CircuitBreakerEvaluation(
                action="TRIP_AND_ROLLBACK",
                step_name=step_name,
                drift_score=drift_score,
                threshold=active_threshold,
                tripped=True,
                circuit_state=self.state.value,
                extracted_claims=claims_dump,
                drift_result=drift_res_dump,
                corrective_directive=corrective_directive,
                rollback_target=rollback_data,
                purged_output=purged_output,
                reason=reason,
            )
        else:
            # VERIFIED
            if self.state == CircuitState.OPEN:
                self.transition_to_half_open("Step passed drift check during recovery")
            elif self.state == CircuitState.HALF_OPEN:
                self.consecutive_successes += 1
                if self.consecutive_successes >= self.recovery_patience:
                    self.reset()
            else:
                self.consecutive_successes += 1
                self.failure_count = 0

            reason = (
                f"Step '{step_name}' verified: epistemic drift D_t={drift_score:.4f} "
                f"< tau={active_threshold:.4f}."
            )

            # Save safe checkpoint
            safe_cp = self.save_checkpoint(
                step_name=step_name,
                state_snapshot=state,
                drift_score=drift_score,
                active_claims=claims_dump,
            )

            eval_record = StepEvaluation(
                step_id=step_name,
                confidence=max(0.0, 1.0 - drift_score),
                entropy=drift_res.semantic_entropy,
                passed=True,
                state_before=state_before,
                state_after=self.state,
                reason=reason,
                metadata={"drift_score": drift_score, "checkpoint_id": safe_cp.checkpoint_id},
            )
            self.history.append(eval_record)

            return CircuitBreakerEvaluation(
                action="VERIFIED",
                step_name=step_name,
                drift_score=drift_score,
                threshold=active_threshold,
                tripped=False,
                circuit_state=self.state.value,
                extracted_claims=claims_dump,
                drift_result=drift_res_dump,
                corrective_directive=None,
                rollback_target=None,
                purged_output=None,
                reason=reason,
            )

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
        Legacy evaluation method for backward compatibility with existing tests and endpoints.
        """
        mock_state = state_snapshot or {}
        drift_score = 1.0 - confidence if confidence is not None else 0.0
        if entropy is not None:
            drift_score = max(drift_score, entropy)

        if content:
            eval_res = self.evaluate_transition(
                state=mock_state,
                step_name=step_id,
                agent_output=content,
            )
            return {
                "step_id": step_id,
                "passed": not eval_res.tripped,
                "confidence": 1.0 - eval_res.drift_score,
                "entropy": eval_res.drift_result.get("semantic_entropy") if eval_res.drift_result else 0.0,
                "threshold": self.threshold,
                "state_before": self.state.value,
                "state_after": eval_res.circuit_state,
                "reason": eval_res.reason,
                "rolled_back": eval_res.tripped,
                "rollback_target": eval_res.rollback_target,
                "failure_count": self.failure_count,
                "consecutive_successes": self.consecutive_successes,
                "drift_score": eval_res.drift_score,
                "action": eval_res.action,
            }

        # Fallback heuristic calculation if content is empty
        passed = confidence >= self.threshold
        reason = f"Epistemic check passed ({confidence:.3f})" if passed else f"Confidence low ({confidence:.3f})"
        if not passed:
            self.trip(reason=reason)
            rb = self.rollback()
        else:
            if state_snapshot:
                self.save_checkpoint(step_id, state_snapshot, drift_score)
            rb = None

        return {
            "step_id": step_id,
            "passed": passed,
            "confidence": confidence,
            "entropy": entropy,
            "threshold": self.threshold,
            "state_before": self.state.value,
            "state_after": self.state.value,
            "reason": reason,
            "rolled_back": not passed,
            "rollback_target": rb,
            "failure_count": self.failure_count,
            "consecutive_successes": self.consecutive_successes,
        }

    def get_status(self) -> Dict[str, Any]:
        """Comprehensive telemetry view of circuit breaker status."""
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
            "total_trips": self.total_trips,
            "total_rollbacks": self.total_rollbacks,
            "total_checkpoints": len(self.checkpoints),
            "history_count": len(self.history),
            "sliding_window_history": [item.model_dump(mode="json") for item in self.history],
            "last_safe_checkpoint": safe_cp.model_dump(mode="json") if safe_cp else None,
            "last_rollback_checkpoint": (
                self.last_rollback_checkpoint.model_dump(mode="json")
                if self.last_rollback_checkpoint
                else None
            ),
        }
