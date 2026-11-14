import asyncio
from enum import Enum
import functools
import json
import logging
import os
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Load environment configuration
load_dotenv()

logger = logging.getLogger("epistemic-circuit-breaker.api")

# Circuit breaker backward-compatible import
try:
    from middleware.circuit_breaker import EpistemicCircuitBreaker
except ImportError:
    from circuit_breaker import EpistemicCircuitBreaker

# Agent pipeline imports
try:
    from agents.pipeline import run_pipeline, simulate_agent_run
except ImportError:
    try:
        from agents import simulate_agent_run
    except ImportError:
        simulate_agent_run = None
    run_pipeline = None

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


def _serialize_payload(data: Any) -> Any:
    """Recursively serializes payloads converting Pydantic models, enums, and dicts to JSON-safe primitives."""
    if data is None:
        return None
    if hasattr(data, "model_dump"):
        return data.model_dump()
    if isinstance(data, Enum):
        return data.value
    if isinstance(data, dict):
        return {str(k): _serialize_payload(v) for k, v in data.items()}
    if isinstance(data, (list, tuple, set)):
        return [_serialize_payload(v) for v in data]
    return data


class EventBroadcaster:
    """
    Publish-Subscribe broadcaster for real-time SSE streaming.
    Allows multiple web clients to listen to live pipeline execution events simultaneously.
    """

    def __init__(self):
        self._subscribers: List[asyncio.Queue] = []
        self._lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            self._subscribers.append(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue) -> None:
        async with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def publish_sync(self, event: str, data: Any) -> None:
        payload = {"event": event, "data": _serialize_payload(data)}
        for q in list(self._subscribers):
            try:
                q.put_nowait(payload)
            except Exception:
                pass

    async def publish(self, event: str, data: Any) -> None:
        payload = {"event": event, "data": _serialize_payload(data)}
        async with self._lock:
            for q in list(self._subscribers):
                try:
                    q.put_nowait(payload)
                except Exception:
                    pass


broadcaster = EventBroadcaster()


# ---------------------------------------------------------------------------
# Pydantic V2 Request / Response Schemas
# ---------------------------------------------------------------------------

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


class RunPipelineRequest(BaseModel):
    task_prompt: str = Field(..., description="The user task prompt or programming requirement")
    constraints: List[str] = Field(default_factory=list, description="System and safety constraints")
    fault_injection_enabled: bool = Field(default=False, description="Whether to inject synthetic hallucination in planner")
    circuit_breaker_enabled: bool = Field(default=True, description="Whether the epistemic circuit breaker is active")
    drift_threshold: float = Field(default=0.50, ge=0.0, le=1.0, description="Epistemic drift trip threshold tau")
    max_retries: int = Field(default=2, ge=0, le=5, description="Maximum rollback retries per stage")
    thread_id: Optional[str] = Field(default=None, description="Optional thread/session ID for checkpointer")


class RunPipelineResponse(BaseModel):
    task_prompt: str
    constraints: List[str] = Field(default_factory=list)
    plan: Optional[str] = None
    generated_code: Optional[str] = None
    review_feedback: Optional[str] = None
    final_code: Optional[str] = None
    step_logs: List[Dict[str, Any]] = Field(default_factory=list)
    drift_history: List[float] = Field(default_factory=list)
    rollback_count: int = 0
    active_claims: List[Dict[str, Any]] = Field(default_factory=list)
    circuit_state: str = "CLOSED"
    last_drift_result: Optional[Dict[str, Any]] = None
    retry_count_planner: int = 0
    retry_count_coder: int = 0
    fault_injection_enabled: bool = False
    circuit_breaker_enabled: bool = True


# ---------------------------------------------------------------------------
# Compatibility & Status Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health_check():
    """Exact requirement health check endpoint."""
    return {"status": "ready", "project": "Epistemic Circuit Breaker"}


@app.get("/api/circuit-breaker/status")
async def get_circuit_breaker_status():
    """Returns current state (CLOSED, OPEN, HALF_OPEN), failure count, and sliding window history."""
    return circuit_breaker.get_status()


@app.post("/api/circuit-breaker/reset")
async def reset_circuit_breaker():
    """Resets the circuit breaker to CLOSED state and clears error counters."""
    circuit_breaker.reset()
    return {
        "status": "reset",
        "message": "Circuit breaker successfully reset to CLOSED state.",
        "telemetry": circuit_breaker.get_status(),
    }


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


@app.post("/api/agent/simulate")
async def simulate_multi_agent_run(req: SimulateRequest):
    """
    Simulates a multi-agent run with optional hallucination injection
    to demonstrate epistemic circuit breaker tripping and state rollback.
    """
    try:
        from agents.pipeline import simulate_agent_run as sim_fn
    except ImportError:
        sim_fn = simulate_agent_run

    if sim_fn is None:
        raise HTTPException(status_code=500, detail="simulate_agent_run is not available")

    try:
        simulation_result = sim_fn(
            task=req.task,
            inject_hallucination=req.inject_hallucination,
            max_iterations=req.max_iterations,
            circuit_breaker=circuit_breaker,
        )
        return simulation_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")


# ---------------------------------------------------------------------------
# Epistemic Circuit Breaker Pipeline Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/run", response_model=RunPipelineResponse)
async def run_pipeline_endpoint(req: RunPipelineRequest):
    """
    Executes the multi-agent epistemic circuit breaker pipeline.
    Invokes run_pipeline(...) from agents.pipeline.
    Returns the final execution state, step logs, drift history, rollback count,
    active claims, and final code.
    Also broadcasts all pipeline lifecycle events to active SSE stream listeners.
    """
    try:
        from agents.pipeline import run_pipeline as runner
    except ImportError:
        runner = run_pipeline

    if runner is None:
        raise HTTPException(
            status_code=501,
            detail="run_pipeline is not available in agents.pipeline",
        )

    # Callback to publish real-time events to all SSE subscribers
    def on_event(event_name: str, event_data: Dict[str, Any]) -> None:
        broadcaster.publish_sync(event_name, event_data)

    try:
        call_fn = functools.partial(
            runner,
            task_prompt=req.task_prompt,
            constraints=req.constraints,
            fault_injection_enabled=req.fault_injection_enabled,
            circuit_breaker_enabled=req.circuit_breaker_enabled,
            drift_threshold=req.drift_threshold,
            max_retries=req.max_retries,
            thread_id=req.thread_id,
            event_callback=on_event,
        )
        loop = asyncio.get_running_loop()
        final_state = await loop.run_in_executor(None, call_fn)
    except Exception as e:
        logger.exception("Pipeline execution failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Pipeline execution error: {str(e)}")

    return RunPipelineResponse(
        task_prompt=final_state.get("task_prompt", req.task_prompt),
        constraints=final_state.get("constraints", req.constraints),
        plan=final_state.get("plan"),
        generated_code=final_state.get("generated_code"),
        review_feedback=final_state.get("review_feedback"),
        final_code=final_state.get("final_code"),
        step_logs=final_state.get("step_logs", []),
        drift_history=final_state.get("drift_history", []),
        rollback_count=final_state.get("rollback_count", 0),
        active_claims=final_state.get("active_claims", []),
        circuit_state=final_state.get("circuit_state", "CLOSED"),
        last_drift_result=final_state.get("last_drift_result"),
        retry_count_planner=final_state.get("retry_count_planner", 0),
        retry_count_coder=final_state.get("retry_count_coder", 0),
        fault_injection_enabled=final_state.get("fault_injection_enabled", req.fault_injection_enabled),
        circuit_breaker_enabled=final_state.get("circuit_breaker_enabled", req.circuit_breaker_enabled),
    )


@app.get("/api/stream")
async def stream_pipeline_endpoint(
    request: Request,
    task_prompt: Optional[str] = Query(None, description="Task prompt to execute immediately and stream"),
    constraints: Optional[str] = Query(None, description="Comma-separated or JSON list of constraints"),
    fault_injection_enabled: bool = Query(False, description="Whether to inject synthetic fault"),
    circuit_breaker_enabled: bool = Query(True, description="Whether circuit breaker is enabled"),
    drift_threshold: float = Query(0.50, description="Epistemic drift trip threshold tau"),
    max_retries: int = Query(2, description="Max rollback retries"),
    thread_id: Optional[str] = Query(None, description="Optional thread/session ID"),
):
    """
    Server-Sent Events (SSE) endpoint: text/event-stream.
    Streams real-time pipeline events formatted as `data: {"event": "...", "data": ...}\n\n`.
    Events emitted:
      - step_start: agent node starting (planner, planner_arbitration, coder, coder_arbitration, reviewer)
      - claims_extracted: atomic claims list
      - drift_evaluation: D_t score, P_contra, H_sem, tripped boolean
      - circuit_trip: emitted when breaker trips with rollback details and corrective directive
      - step_complete: agent node completed with output
      - pipeline_complete: final code and metrics

    Modes:
      1. Triggered Execution: If `task_prompt` query parameter is provided, executes the pipeline
         and streams all execution events directly until completion.
      2. Event Listener: If `task_prompt` is omitted, subscribes to the shared event broadcaster,
         streaming real-time events triggered by other clients (e.g. POST /api/run) with periodic keep-alive pings.
    """
    parsed_constraints: List[str] = []
    if constraints:
        try:
            loaded = json.loads(constraints)
            if isinstance(loaded, list):
                parsed_constraints = [str(x) for x in loaded]
            else:
                parsed_constraints = [str(loaded)]
        except Exception:
            parsed_constraints = [c.strip() for c in constraints.split(",") if c.strip()]

    # Mode 1: Direct Pipeline Execution and SSE Streaming
    if task_prompt:
        try:
            from agents.pipeline import run_pipeline as runner
        except ImportError:
            runner = run_pipeline

        if runner is None:
            raise HTTPException(
                status_code=501,
                detail="run_pipeline is not available in agents.pipeline",
            )

        client_queue: asyncio.Queue = asyncio.Queue()

        def stream_callback(event_name: str, event_data: Dict[str, Any]) -> None:
            serialized = _serialize_payload(event_data)
            payload = {"event": event_name, "data": serialized}
            try:
                client_queue.put_nowait(payload)
            except Exception:
                pass
            broadcaster.publish_sync(event_name, serialized)

        async def run_generator() -> AsyncGenerator[str, None]:
            loop = asyncio.get_running_loop()
            call_fn = functools.partial(
                runner,
                task_prompt=task_prompt,
                constraints=parsed_constraints,
                fault_injection_enabled=fault_injection_enabled,
                circuit_breaker_enabled=circuit_breaker_enabled,
                drift_threshold=drift_threshold,
                max_retries=max_retries,
                thread_id=thread_id,
                event_callback=stream_callback,
            )
            pipeline_future = loop.run_in_executor(None, call_fn)

            done = False
            while not done:
                if await request.is_disconnected():
                    logger.info("SSE client disconnected during stream execution.")
                    break

                try:
                    item = await asyncio.wait_for(client_queue.get(), timeout=1.0)
                    yield f"data: {json.dumps(item)}\n\n"
                    if item.get("event") == "pipeline_complete":
                        done = True
                except asyncio.TimeoutError:
                    if pipeline_future.done():
                        # Drain any pending items before terminating
                        while not client_queue.empty():
                            try:
                                pending_item = client_queue.get_nowait()
                                yield f"data: {json.dumps(pending_item)}\n\n"
                            except Exception:
                                break
                        exc = pipeline_future.exception()
                        if exc:
                            err_item = {"event": "error", "data": {"error": str(exc)}}
                            yield f"data: {json.dumps(err_item)}\n\n"
                        done = True
                    else:
                        yield ": ping\n\n"

            # Drain any remaining events in queue
            while not client_queue.empty():
                try:
                    item = client_queue.get_nowait()
                    yield f"data: {json.dumps(item)}\n\n"
                except Exception:
                    break

        return StreamingResponse(
            run_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Mode 2: Real-time Telemetry Listener (Subscribes to Event Broadcaster)
    async def listen_generator() -> AsyncGenerator[str, None]:
        listener_queue = await broadcaster.subscribe()
        try:
            init_item = {
                "event": "connected",
                "data": {"message": "Subscribed to Epistemic Circuit Breaker SSE telemetry stream."},
            }
            yield f"data: {json.dumps(init_item)}\n\n"

            while True:
                if await request.is_disconnected():
                    break
                try:
                    item = await asyncio.wait_for(listener_queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(item)}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            await broadcaster.unsubscribe(listener_queue)

    return StreamingResponse(
        listen_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=True)
