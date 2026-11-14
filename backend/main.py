import os
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from circuit_breaker import EpistemicCircuitBreaker
from agents import simulate_agent_run

# Load environment configuration
load_dotenv()

app = FastAPI(
    title="Epistemic Circuit Breaker API",
    description="Research prototype backend for Epistemic Circuit Breakers in Autonomous Multi-Agent Workflows",
    version="1.0.0",
)

# CORS Middleware enabled for localhost and flexible development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared global instance of the Epistemic Circuit Breaker
circuit_breaker = EpistemicCircuitBreaker()


class EvaluateStepRequest(BaseModel):
    step_id: str = Field(..., description="Unique identifier for the agent step")
    content: Optional[str] = Field("", description="Generated reasoning, claim, or action payload")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Epistemic confidence score (0.0 - 1.0)")
    entropy: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional uncertainty entropy score")
    state_snapshot: Optional[Dict[str, Any]] = Field(None, description="Current agent memory or state to checkpoint")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context metadata")


class SimulateRequest(BaseModel):
    task: str = Field(
        default="Formulate an empirical theorem on epistemic stability in autonomous multi-agent networks.",
        description="The research or reasoning task for the agents",
    )
    inject_hallucination: bool = Field(
        default=False,
        description="Whether to deliberately inject high-entropy hallucinated claims to test circuit tripping and rollback",
    )
    max_iterations: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry iterations allowed upon rollback",
    )


@app.get("/api/health")
async def health_check():
    """Exact requirement health check endpoint."""
    return {"status": "ready", "project": "Epistemic Circuit Breaker"}


@app.get("/api/circuit-breaker/status")
async def get_circuit_breaker_status():
    """Returns current state (CLOSED, OPEN, HALF_OPEN), failure count, and sliding window history."""
    return circuit_breaker.get_status()


@app.post("/api/circuit-breaker/evaluate")
async def evaluate_step(req: EvaluateStepRequest):
    """
    Evaluates epistemic confidence of an agent step and trips or rolls back
    if hallucination entropy threshold is exceeded.
    """
    result = circuit_breaker.evaluate_step(
        step_id=req.step_id,
        content=req.content or "",
        confidence=req.confidence,
        entropy=req.entropy,
        state_snapshot=req.state_snapshot,
        metadata=req.metadata,
    )
    return result


@app.post("/api/circuit-breaker/reset")
async def reset_circuit_breaker():
    """Resets the circuit breaker to CLOSED state and clears error counters."""
    circuit_breaker.reset()
    return {
        "status": "reset",
        "message": "Circuit breaker successfully reset to CLOSED state.",
        "telemetry": circuit_breaker.get_status(),
    }


@app.post("/api/agent/simulate")
async def simulate_multi_agent_run(req: SimulateRequest):
    """
    Simulates a multi-agent run with optional hallucination injection
    to demonstrate epistemic circuit breaker tripping and state rollback.
    """
    try:
        simulation_result = simulate_agent_run(
            task=req.task,
            inject_hallucination=req.inject_hallucination,
            max_iterations=req.max_iterations,
            circuit_breaker=circuit_breaker,
        )
        return simulation_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=True)
