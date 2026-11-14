"""
Epistemic Circuit Breaker - Module 3 Middleware
Provides core NLP claim extraction, epistemic drift scoring, and circuit breaker arbitration.
"""

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
from .circuit_breaker import (
    Checkpoint,
    CircuitBreakerEvaluation,
    CircuitState,
    EpistemicCircuitBreaker,
    StepEvaluation,
)

__all__ = [
    "AtomicClaim",
    "ClaimExtractionResult",
    "ClaimType",
    "extract_atomic_claims",
    "ClaimEvaluation",
    "DriftEvaluationResult",
    "EpistemicDriftScorer",
    "get_drift_scorer",
    "Checkpoint",
    "CircuitBreakerEvaluation",
    "CircuitState",
    "EpistemicCircuitBreaker",
    "StepEvaluation",
]
