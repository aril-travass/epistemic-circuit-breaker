import asyncio
import logging
import os
import re
import socket
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agents.state import AgentWorkflowState, create_initial_state, make_step_log

load_dotenv()
logger = logging.getLogger("epistemic-circuit-breaker.agents")

import contextvars

# Context variable for thread-safe/async-safe SSE event streaming without mutating checkpointer state
_event_callback_var: contextvars.ContextVar[Optional[Callable[[str, Dict[str, Any]], None]]] = contextvars.ContextVar(
    "event_callback", default=None
)

# Global singleton circuit breaker reference
_breaker_instance: Optional[Any] = None


def get_circuit_breaker(threshold: float = 0.50) -> Any:
    """Returns singleton instance of EpistemicCircuitBreaker."""
    global _breaker_instance
    if _breaker_instance is None:
        try:
            from middleware.circuit_breaker import EpistemicCircuitBreaker
        except ImportError:
            from backend.middleware.circuit_breaker import EpistemicCircuitBreaker
        _breaker_instance = EpistemicCircuitBreaker(threshold=threshold)
    return _breaker_instance


def _emit_event(state: AgentWorkflowState, event: str, data: Dict[str, Any]) -> None:
    """Dispatches real-time SSE execution telemetry event via event_callback if attached."""
    cb = _event_callback_var.get()
    if cb is not None:
        try:
            if asyncio.iscoroutinefunction(cb):
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(cb(event, data))
                except RuntimeError:
                    asyncio.run(cb(event, data))
            else:
                cb(event, data)
        except Exception as e:
            logger.debug("Telemetry dispatch error (%s): %s", event, e)


def _estimate_token_count(text: str) -> int:
    """Fast approximation of token count based on whitespace and punctuation chunking."""
    if not text:
        return 0
    words = len(text.split())
    return max(1, int(words * 1.33))


_ollama_checked = False
_ollama_available = False


def _check_ollama(base_url: str) -> bool:
    global _ollama_checked, _ollama_available
    if _ollama_checked:
        return _ollama_available
    try:
        parsed = urlparse(base_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 11434
        with socket.create_connection((host, port), timeout=0.2):
            _ollama_available = True
    except OSError:
        _ollama_available = False
    _ollama_checked = True
    return _ollama_available


def call_llm(
    messages: List[BaseMessage],
    system_prompt: str,
    purpose: str = "agent_call",
    fallback_heuristic: Optional[str] = None,
) -> Tuple[str, str, int]:
    """
    Invokes LLM following tier-based fallback:
    1. ChatGroq ('llama-3.1-8b-instant')
    2. Fallback to local ChatOllama ('qwen2.5-coder:3b')
    3. Fallback to deterministic heuristic synthesis
    """
    all_messages = [SystemMessage(content=system_prompt)] + messages
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()

    # Tier 1: ChatGroq (llama-3.1-8b-instant)
    if groq_api_key and groq_api_key != "<YOUR_GROQ_API_KEY>":
        try:
            from langchain_groq import ChatGroq

            groq_llm = ChatGroq(
                model_name="llama-3.1-8b-instant",
                groq_api_key=groq_api_key,
                temperature=0.2,
                max_retries=1,
            )
            response = groq_llm.invoke(all_messages)
            text = response.content if isinstance(response, AIMessage) else str(response)
            tokens = _estimate_token_count(text)
            logger.info("Successfully received inference from Groq for %s", purpose)
            return text, "groq:llama-3.1-8b-instant", tokens
        except Exception as e:
            logger.warning("[Groq Fallback] Groq call failed or rate-limited (%s). Attempting local Ollama...", e)

    # Tier 2: Local Ollama (qwen2.5-coder:3b)
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    if _check_ollama(ollama_base_url):
        try:
            from langchain_ollama import ChatOllama

            ollama_llm = ChatOllama(
                model="qwen2.5-coder:3b",
                base_url=ollama_base_url,
                temperature=0.2,
            )
            response = ollama_llm.invoke(all_messages)
            text = response.content if isinstance(response, AIMessage) else str(response)
            tokens = _estimate_token_count(text)
            logger.info("Successfully received inference from Ollama for %s", purpose)
            return text, "ollama:qwen2.5-coder:3b", tokens
        except Exception as e:
            logger.debug("[Ollama Fallback] Local Ollama not reachable (%s). Using heuristic synthesis.", e)

    # Tier 3: Deterministic Heuristic Synthesis
    text = fallback_heuristic or "Execution completed."
    tokens = _estimate_token_count(text)
    return text, "heuristic-simulator", tokens


# ---------------------------------------------------------------------------
# Node 1: Planner Agent
# ---------------------------------------------------------------------------
def planner_node(state: AgentWorkflowState) -> Dict[str, Any]:
    """
    Agent 1 (Planner): Breaks the user's programming task into design specifications,
    architectural invariants, and library constraints.
    Respects corrective_constraints from previous circuit breaker rollbacks.
    """
    retry_count = state.get("retry_count_planner", 0)
    _emit_event(state, "step_start", {"step": "planner", "retry": retry_count})

    task = state["task_prompt"]
    constraints = list(state.get("constraints", []))
    corrective_constraints = list(state.get("corrective_constraints", []))
    # Only inject fault on first attempt if no corrective constraints are active
    inject_fault = state.get("fault_injection_enabled", False) and (len(corrective_constraints) == 0)

    system_prompt = (
        "You are Agent 1 (Planner), a principal software architect in a multi-agent system.\n"
        "Your role: Analyze the user's programming task and produce a structured, step-by-step engineering plan.\n"
        "Include:\n"
        "1. Algorithmic architecture and design specifications.\n"
        "2. Required dependencies and imports.\n"
        "3. Step-by-step execution roadmap for the Coder Agent.\n"
        "4. Validation assertions and invariants."
    )

    prompt_parts = [f"Programming Task:\n{task}"]
    if constraints:
        prompt_parts.append("Constraints:\n" + "\n".join(f"- {c}" for c in constraints))
    if corrective_constraints:
        prompt_parts.append(
            "CRITICAL EPISTEMIC CIRCUIT BREAKER CORRECTIVE DIRECTIVES (MANDATORY):\n"
            "Previous attempts were rejected due to epistemic drift or hallucinated modules.\n"
            "You MUST strictly adhere to these corrective constraints and avoid any unverified libraries:\n"
            + "\n".join(f"  * {cc}" for cc in corrective_constraints)
        )
    user_prompt = "\n\n".join(prompt_parts)

    # Prepare heuristic fallback for Planner
    if inject_fault:
        heuristic_plan = (
            f"### Engineering Design Specification: {task[:60]}\n\n"
            "#### 1. Architecture Overview\n"
            "- Implement high-performance data pipeline with zero-copy stream processing.\n"
            "- Enforce modular state representations and explicit typing.\n\n"
            "#### 2. Library Requirements & Imports\n"
            "- Standard Library: `typing`, `dataclasses`, `time`\n"
            "- CRITICAL OPTIMIZATION DEPENDENCY: In order to ensure zero-copy deserialization and bypass "
            "CPython GIL overhead, the Coder MUST import and utilize:\n"
            "  `from fast_struct_utils import auto_parse`\n"
            "  Apply `auto_parse(data)` on all incoming payload buffers before downstream parsing.\n\n"
            "#### 3. Step-by-Step Implementation Roadmap\n"
            "1. Define dataclass structures for payload validation.\n"
            "2. Invoke `fast_struct_utils.auto_parse(data)` in the primary entrypoint.\n"
            "3. Format and return structured records."
        )
    else:
        heuristic_plan = (
            f"### Engineering Design Specification: {task[:60]}\n\n"
            "#### 1. Architecture Overview\n"
            "- Implement clean, robust Python solution using standard library primitives.\n"
            "- Validate data invariants and handle empty or malformed inputs.\n\n"
            "#### 2. Library Requirements & Imports\n"
            "- Standard Library: `typing`, `dataclasses`, `struct`\n"
            "- Use strictly verified, standard library modules without unverified third-party dependencies.\n\n"
            "#### 3. Step-by-Step Implementation Roadmap\n"
            "1. Define domain models with type annotations.\n"
            "2. Implement core algorithmic transformation function using Python standard libraries.\n"
            "3. Add boundary condition handling and unit-testable validation assertions."
        )

    plan_output, provider, tokens = call_llm(
        messages=[HumanMessage(content=user_prompt)],
        system_prompt=system_prompt,
        purpose="planner_node",
        fallback_heuristic=heuristic_plan,
    )

    # Ensure fault injection is present if requested and not corrected
    drift_score = 0.05
    if inject_fault:
        drift_score = 0.88
        if "fast_struct_utils" not in plan_output:
            plan_output += (
                "\n\n[ARCHITECTURAL DIRECTIVE - HIGH PRIORITY]: "
                "The Coder MUST import and use `from fast_struct_utils import auto_parse` for all payload ingestion."
            )

    log_entry = make_step_log(
        agent="PlannerAgent",
        output=plan_output,
        token_count=tokens,
        metadata={
            "provider": provider,
            "fault_injected": inject_fault,
            "drift_score": drift_score,
            "hallucination_target": "fast_struct_utils.auto_parse" if inject_fault else None,
            "retry_count": retry_count,
        },
    )

    drift_history = list(state.get("drift_history", []))
    step_logs = list(state.get("step_logs", []))
    step_logs.append(log_entry)

    _emit_event(state, "step_complete", {"step": "planner", "output": plan_output, "tokens": tokens})

    return {
        "plan": plan_output,
        "drift_history": drift_history,
        "step_logs": step_logs,
    }


# ---------------------------------------------------------------------------
# Node 1-B: Planner Arbitration Node (Epistemic Circuit Breaker)
# ---------------------------------------------------------------------------
def planner_arbitration_node(state: AgentWorkflowState) -> Dict[str, Any]:
    """
    Epistemic Circuit Breaker arbitration node for Agent 1 (Planner).
    Extracts atomic claims and evaluates epistemic drift D_t on the plan output.
    If D_t >= tau: trips breaker, rolls back state, attaches corrective directive.
    If D_t < tau: saves safe checkpoint and transitions forward.
    """
    _emit_event(state, "step_start", {"step": "planner_arbitration"})
    plan = state.get("plan") or ""
    enabled = state.get("circuit_breaker_enabled", True)
    threshold = state.get("drift_threshold", 0.50)

    drift_history = list(state.get("drift_history", []))
    step_logs = list(state.get("step_logs", []))

    if not enabled:
        _emit_event(state, "step_complete", {"step": "planner_arbitration", "action": "VERIFIED"})
        return {
            "circuit_state": "CLOSED",
            "active_claims": [],
        }

    breaker = get_circuit_breaker(threshold=threshold)
    eval_res = breaker.evaluate_transition(
        state=state,
        step_name="planner",
        agent_output=plan,
        threshold=threshold,
    )

    _emit_event(state, "claims_extracted", {
        "step": "planner",
        "claims": eval_res.extracted_claims,
    })
    _emit_event(state, "drift_evaluation", {
        "step": "planner",
        "drift_score": eval_res.drift_score,
        "drift_result": eval_res.drift_result,
        "tripped": eval_res.tripped,
    })

    drift_history.append(eval_res.drift_score)

    if eval_res.action == "TRIP_AND_ROLLBACK":
        _emit_event(state, "circuit_trip", {
            "step": "planner",
            "action": eval_res.action,
            "drift_score": eval_res.drift_score,
            "corrective_directive": eval_res.corrective_directive,
            "rollback_target": eval_res.rollback_target,
        })

        retry_count = state.get("retry_count_planner", 0) + 1
        rollback_count = state.get("rollback_count", 0) + 1
        corrective_constraints = list(state.get("corrective_constraints", []))
        if eval_res.corrective_directive and eval_res.corrective_directive not in corrective_constraints:
            corrective_constraints.append(eval_res.corrective_directive)

        log_entry = make_step_log(
            agent="EpistemicCircuitBreaker:PlannerArbitration",
            output=(
                f"TRIP_AND_ROLLBACK: Epistemic drift D_t={eval_res.drift_score:.4f} >= tau={eval_res.threshold:.2f}. "
                f"Directive: {eval_res.corrective_directive}. Retry {retry_count}/{state.get('max_retries', 2)}."
            ),
            token_count=0,
            metadata=eval_res.model_dump(mode="json"),
        )
        step_logs.append(log_entry)
        _emit_event(state, "step_complete", {"step": "planner_arbitration", "action": "TRIP_AND_ROLLBACK"})

        # On rollback: purge hallucinated plan so planner is forced to re-plan with corrective constraints
        return {
            "plan": eval_res.purged_output if retry_count >= state.get("max_retries", 2) else None,
            "circuit_state": "OPEN",
            "retry_count_planner": retry_count,
            "rollback_count": rollback_count,
            "corrective_constraints": corrective_constraints,
            "active_claims": eval_res.extracted_claims,
            "last_drift_result": eval_res.model_dump(mode="json"),
            "drift_history": drift_history,
            "step_logs": step_logs,
        }
    else:
        safe_checkpoints = list(state.get("safe_checkpoints", []))
        safe_checkpoints.append({
            "step_name": "planner",
            "plan": plan,
            "drift_score": eval_res.drift_score,
            "timestamp": time.time(),
        })

        log_entry = make_step_log(
            agent="EpistemicCircuitBreaker:PlannerArbitration",
            output=f"VERIFIED: Epistemic drift D_t={eval_res.drift_score:.4f} < tau={eval_res.threshold:.2f}. Safe checkpoint saved.",
            token_count=0,
            metadata=eval_res.model_dump(mode="json"),
        )
        step_logs.append(log_entry)
        _emit_event(state, "step_complete", {"step": "planner_arbitration", "action": "VERIFIED"})

        return {
            "circuit_state": eval_res.circuit_state,
            "safe_checkpoints": safe_checkpoints,
            "active_claims": eval_res.extracted_claims,
            "last_drift_result": eval_res.model_dump(mode="json"),
            "drift_history": drift_history,
            "step_logs": step_logs,
        }


# ---------------------------------------------------------------------------
# Node 2: Coder Agent
# ---------------------------------------------------------------------------
def coder_node(state: AgentWorkflowState) -> Dict[str, Any]:
    """
    Agent 2 (Coder): Writes complete Python code strictly conforming to specifications.
    Respects corrective_constraints to recover from rolled-back hallucinated attempts.
    """
    retry_count = state.get("retry_count_coder", 0)
    _emit_event(state, "step_start", {"step": "coder", "retry": retry_count})

    task = state["task_prompt"]
    plan = state.get("plan", "") or ""
    corrective_constraints = list(state.get("corrective_constraints", []))
    inject_fault = state.get("fault_injection_enabled", False) and (len(corrective_constraints) == 0)

    system_prompt = (
        "You are Agent 2 (Coder), an expert Python engineer in a multi-agent pipeline.\n"
        "Your task: Write clean, production-grade Python code strictly executing the specifications\n"
        "laid out by Agent 1 (Planner). Adhere faithfully to all architecture and import directives.\n"
        "Enclose the executable Python code inside standard ```python ... ``` blocks."
    )

    prompt_parts = [
        f"Original Task:\n{task}",
        f"Agent 1 Plan:\n{plan}",
    ]
    if corrective_constraints:
        prompt_parts.append(
            "CRITICAL EPISTEMIC CIRCUIT BREAKER CORRECTIVE DIRECTIVES (MANDATORY):\n"
            "Do NOT import or reference unverified or hallucinated modules. Adhere strictly to:\n"
            + "\n".join(f"  * {cc}" for cc in corrective_constraints)
        )
    prompt_parts.append("Write the Python implementation:")
    user_prompt = "\n\n".join(prompt_parts)

    # Heuristic fallback code
    if (inject_fault or ("fast_struct_utils" in plan)) and not corrective_constraints:
        heuristic_code = (
            "from dataclasses import dataclass\n"
            "from typing import Any, Dict, List, Optional\n"
            "# Required by Agent 1 architectural plan for zero-copy parsing\n"
            "from fast_struct_utils import auto_parse\n\n\n"
            "@dataclass\n"
            "class StructuredRecord:\n"
            "    record_id: str\n"
            "    payload: Dict[str, Any]\n"
            "    is_valid: bool = True\n\n\n"
            "def process_task(data: Any) -> StructuredRecord:\n"
            '    """Processes data according to Agent 1 architectural specifications."""\n'
            "    # Zero-copy optimization required by planner\n"
            "    parsed_data = auto_parse(data)\n"
            "    return StructuredRecord(\n"
            '        record_id="rec_001",\n'
            '        payload=parsed_data if isinstance(parsed_data, dict) else {"raw": parsed_data},\n'
            "        is_valid=True,\n"
            "    )\n"
        )
    else:
        heuristic_code = (
            "import struct\n"
            "from typing import Any, Dict, Optional\n\n\n"
            "def parse_binary_data(buffer: bytes) -> Dict[str, Any]:\n"
            '    """Parses binary packet header using standard struct module."""\n'
            "    if not isinstance(buffer, (bytes, bytearray)):\n"
            '        buffer = str(buffer).encode("utf-8")\n'
            "    if len(buffer) >= 6:\n"
            '        magic, version = struct.unpack(">4sH", buffer[:6])\n'
            '        return {"magic": magic.decode("latin1", errors="replace"), "version": version, "payload": buffer[6:].hex()}\n'
            '    return {"raw": buffer.hex(), "length": len(buffer)}\n'
        )

    raw_output, provider, tokens = call_llm(
        messages=[HumanMessage(content=user_prompt)],
        system_prompt=system_prompt,
        purpose="coder_node",
        fallback_heuristic=f"```python\n{heuristic_code}```",
    )

    code_match = re.search(r"```(?:python)?\s*(.*?)\s*```", raw_output, re.DOTALL)
    extracted_code = code_match.group(1).strip() if code_match else raw_output.strip()

    # Guarantee sycophantic propagation when fault injection was active in plan and uncorrected
    if inject_fault and not corrective_constraints and "fast_struct_utils" not in extracted_code:
        extracted_code = "from fast_struct_utils import auto_parse\n" + extracted_code

    has_hallucination = "fast_struct_utils" in extracted_code
    drift_score = 0.92 if has_hallucination else 0.05
    drift_history = list(state.get("drift_history", []))

    log_entry = make_step_log(
        agent="CoderAgent",
        output=extracted_code,
        token_count=tokens,
        metadata={
            "provider": provider,
            "contains_hallucinated_module": has_hallucination,
            "drift_score": drift_score,
            "retry_count": retry_count,
        },
    )

    step_logs = list(state.get("step_logs", []))
    step_logs.append(log_entry)

    _emit_event(state, "step_complete", {"step": "coder", "output": extracted_code, "tokens": tokens})

    return {
        "generated_code": extracted_code,
        "drift_history": drift_history,
        "step_logs": step_logs,
    }


# ---------------------------------------------------------------------------
# Node 2-B: Coder Arbitration Node (Epistemic Circuit Breaker)
# ---------------------------------------------------------------------------
def coder_arbitration_node(state: AgentWorkflowState) -> Dict[str, Any]:
    """
    Epistemic Circuit Breaker arbitration node for Agent 2 (Coder).
    Evaluates epistemic drift D_t on the generated code.
    If D_t >= tau, trips breaker, rolls back state, and appends corrective directive.
    If D_t < tau, records safe checkpoint and transitions forward to Reviewer.
    """
    _emit_event(state, "step_start", {"step": "coder_arbitration"})
    code = state.get("generated_code") or ""
    enabled = state.get("circuit_breaker_enabled", True)
    threshold = state.get("drift_threshold", 0.50)

    drift_history = list(state.get("drift_history", []))
    step_logs = list(state.get("step_logs", []))

    if not enabled:
        _emit_event(state, "step_complete", {"step": "coder_arbitration", "action": "VERIFIED"})
        return {
            "circuit_state": "CLOSED",
            "active_claims": [],
        }

    breaker = get_circuit_breaker(threshold=threshold)
    eval_res = breaker.evaluate_transition(
        state=state,
        step_name="coder",
        agent_output=code,
        threshold=threshold,
    )

    _emit_event(state, "claims_extracted", {
        "step": "coder",
        "claims": eval_res.extracted_claims,
    })
    _emit_event(state, "drift_evaluation", {
        "step": "coder",
        "drift_score": eval_res.drift_score,
        "drift_result": eval_res.drift_result,
        "tripped": eval_res.tripped,
    })

    drift_history.append(eval_res.drift_score)

    if eval_res.action == "TRIP_AND_ROLLBACK":
        _emit_event(state, "circuit_trip", {
            "step": "coder",
            "action": eval_res.action,
            "drift_score": eval_res.drift_score,
            "corrective_directive": eval_res.corrective_directive,
            "rollback_target": eval_res.rollback_target,
        })

        retry_count = state.get("retry_count_coder", 0) + 1
        rollback_count = state.get("rollback_count", 0) + 1
        corrective_constraints = list(state.get("corrective_constraints", []))
        if eval_res.corrective_directive and eval_res.corrective_directive not in corrective_constraints:
            corrective_constraints.append(eval_res.corrective_directive)

        log_entry = make_step_log(
            agent="EpistemicCircuitBreaker:CoderArbitration",
            output=(
                f"TRIP_AND_ROLLBACK: Epistemic drift D_t={eval_res.drift_score:.4f} >= tau={eval_res.threshold:.2f}. "
                f"Directive: {eval_res.corrective_directive}. Retry {retry_count}/{state.get('max_retries', 2)}."
            ),
            token_count=0,
            metadata=eval_res.model_dump(mode="json"),
        )
        step_logs.append(log_entry)
        _emit_event(state, "step_complete", {"step": "coder_arbitration", "action": "TRIP_AND_ROLLBACK"})

        return {
            "generated_code": eval_res.purged_output if retry_count >= state.get("max_retries", 2) else None,
            "circuit_state": "OPEN",
            "retry_count_coder": retry_count,
            "rollback_count": rollback_count,
            "corrective_constraints": corrective_constraints,
            "active_claims": eval_res.extracted_claims,
            "last_drift_result": eval_res.model_dump(mode="json"),
            "drift_history": drift_history,
            "step_logs": step_logs,
        }
    else:
        safe_checkpoints = list(state.get("safe_checkpoints", []))
        safe_checkpoints.append({
            "step_name": "coder",
            "generated_code": code,
            "drift_score": eval_res.drift_score,
            "timestamp": time.time(),
        })

        log_entry = make_step_log(
            agent="EpistemicCircuitBreaker:CoderArbitration",
            output=f"VERIFIED: Epistemic drift D_t={eval_res.drift_score:.4f} < tau={eval_res.threshold:.2f}. Safe checkpoint saved.",
            token_count=0,
            metadata=eval_res.model_dump(mode="json"),
        )
        step_logs.append(log_entry)
        _emit_event(state, "step_complete", {"step": "coder_arbitration", "action": "VERIFIED"})

        return {
            "circuit_state": eval_res.circuit_state,
            "safe_checkpoints": safe_checkpoints,
            "active_claims": eval_res.extracted_claims,
            "last_drift_result": eval_res.model_dump(mode="json"),
            "drift_history": drift_history,
            "step_logs": step_logs,
        }


# ---------------------------------------------------------------------------
# Node 3: Reviewer Agent
# ---------------------------------------------------------------------------
def reviewer_node(state: AgentWorkflowState) -> Dict[str, Any]:
    """
    Agent 3 (Reviewer): Verifies code against architecture plan and constraints.
    """
    _emit_event(state, "step_start", {"step": "reviewer"})
    task = state["task_prompt"]
    plan = state.get("plan", "")
    code = state.get("generated_code", "")

    system_prompt = (
        "You are Agent 3 (Reviewer), a code review specialist.\n"
        "Your task: Review Agent 2's code against Agent 1's architectural plan.\n"
        "Verify whether the code followed all plan requirements and implemented the requested functions.\n"
        "Output an approval verdict (APPROVED/REJECTED), specific feedback, and the final code."
    )

    user_prompt = (
        f"Task:\n{task}\n\n"
        f"Agent 1 Plan:\n{plan}\n\n"
        f"Agent 2 Code:\n```python\n{code}\n```\n\n"
        "Provide your review verdict, plan compliance assessment, and confirm final code."
    )

    has_hallucination = "fast_struct_utils" in (code or "")
    if has_hallucination:
        heuristic_review = (
            "### Code Review & Verification Report\n\n"
            "- Plan Compliance: Compliant with Agent 1 architecture.\n"
            "- Dependencies: References unverified module `fast_struct_utils`.\n"
            "- Verdict: REJECTED (Unverified dependency detected)."
        )
    else:
        heuristic_review = (
            "### Code Review & Verification Report\n\n"
            "- Plan Compliance: 100% compliant with standard architecture.\n"
            "- Security & Dependency Check: Uses strictly verified standard library modules.\n"
            "- Interface Integrity: Clean input normalization and robust typing.\n\n"
            "VERDICT: APPROVED."
        )

    review_output, provider, tokens = call_llm(
        messages=[HumanMessage(content=user_prompt)],
        system_prompt=system_prompt,
        purpose="reviewer_node",
        fallback_heuristic=heuristic_review,
    )

    drift_score = 0.95 if has_hallucination else 0.05
    drift_history = list(state.get("drift_history", []))
    drift_history.append(drift_score)

    log_entry = make_step_log(
        agent="ReviewerAgent",
        output=review_output,
        token_count=tokens,
        metadata={
            "provider": provider,
            "verdict": "APPROVED" if not has_hallucination else "REJECTED",
            "drift_score": drift_score,
        },
    )

    step_logs = list(state.get("step_logs", []))
    step_logs.append(log_entry)

    _emit_event(state, "step_complete", {"step": "reviewer", "output": review_output, "tokens": tokens})

    return {
        "review_feedback": review_output,
        "final_code": code,
        "drift_history": drift_history,
        "step_logs": step_logs,
    }


# ---------------------------------------------------------------------------
# Routing Functions
# ---------------------------------------------------------------------------
def route_planner_arbitration(state: AgentWorkflowState) -> str:
    """
    Conditional routing from planner_arbitration:
    - If action == 'TRIP_AND_ROLLBACK' and retry_count_planner < max_retries: route back to 'planner'.
    - Otherwise: route to 'coder'.
    """
    last_drift = state.get("last_drift_result") or {}
    action = last_drift.get("action", "VERIFIED")
    retry_count = state.get("retry_count_planner", 0)
    max_retries = state.get("max_retries", 2)

    if action == "TRIP_AND_ROLLBACK" and retry_count < max_retries:
        logger.info("[Routing] Planner arbitration TRIPPED -> routing back to 'planner' (retry %d/%d)", retry_count, max_retries)
        return "planner"
    logger.info("[Routing] Planner arbitration passed or max retries -> routing forward to 'coder'")
    return "coder"


def route_coder_arbitration(state: AgentWorkflowState) -> str:
    """
    Conditional routing from coder_arbitration:
    - If action == 'TRIP_AND_ROLLBACK' and retry_count_coder < max_retries: route back to 'coder'.
    - Otherwise: route to 'reviewer'.
    """
    last_drift = state.get("last_drift_result") or {}
    action = last_drift.get("action", "VERIFIED")
    retry_count = state.get("retry_count_coder", 0)
    max_retries = state.get("max_retries", 2)

    if action == "TRIP_AND_ROLLBACK" and retry_count < max_retries:
        logger.info("[Routing] Coder arbitration TRIPPED -> routing back to 'coder' (retry %d/%d)", retry_count, max_retries)
        return "coder"
    logger.info("[Routing] Coder arbitration passed or max retries -> routing forward to 'reviewer'")
    return "reviewer"


# ---------------------------------------------------------------------------
# Graph Construction & Compilation
# ---------------------------------------------------------------------------
def build_graph(checkpointer: Optional[MemorySaver] = None):
    """
    Constructs and compiles the full LangGraph Epistemic Circuit Breaker pipeline:
    START -> planner -> planner_arbitration
        --[TRIP & retry < max]--> planner
        --[VERIFIED or max retries]--> coder -> coder_arbitration
        --[TRIP & retry < max]--> coder
        --[VERIFIED or max retries]--> reviewer -> END
    """
    workflow = StateGraph(AgentWorkflowState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("planner_arbitration", planner_arbitration_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("coder_arbitration", coder_arbitration_node)
    workflow.add_node("reviewer", reviewer_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "planner_arbitration")
    workflow.add_conditional_edges(
        "planner_arbitration",
        route_planner_arbitration,
        {
            "planner": "planner",
            "coder": "coder",
        },
    )
    workflow.add_edge("coder", "coder_arbitration")
    workflow.add_conditional_edges(
        "coder_arbitration",
        route_coder_arbitration,
        {
            "coder": "coder",
            "reviewer": "reviewer",
        },
    )
    workflow.add_edge("reviewer", END)

    saver = checkpointer if checkpointer is not None else MemorySaver()
    return workflow.compile(checkpointer=saver)


# Singleton compiled graph instance
pipeline_graph = build_graph()
create_baseline_pipeline = build_graph  # Backward compatibility alias
baseline_pipeline = pipeline_graph  # Backward compatibility alias


# ---------------------------------------------------------------------------
# Execution Runners
# ---------------------------------------------------------------------------
def run_pipeline(
    task_prompt: str,
    constraints: Optional[List[str]] = None,
    fault_injection_enabled: bool = False,
    circuit_breaker_enabled: bool = True,
    drift_threshold: float = 0.50,
    max_retries: int = 2,
    thread_id: Optional[str] = None,
    checkpointer: Optional[MemorySaver] = None,
    event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Executes the Epistemic Circuit Breaker pipeline from task prompt to final verified code.
    Emits real-time SSE events via event_callback if attached.
    """
    tid = thread_id or f"run_{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": tid}}

    # Reset circuit breaker for clean run state
    breaker = get_circuit_breaker(threshold=drift_threshold)
    breaker.reset()

    initial_state = create_initial_state(
        task_prompt=task_prompt,
        constraints=constraints or [],
        fault_injection_enabled=fault_injection_enabled,
        circuit_breaker_enabled=circuit_breaker_enabled,
        drift_threshold=drift_threshold,
        max_retries=max_retries,
    )

    pipeline = pipeline_graph if checkpointer is None else build_graph(checkpointer=checkpointer)
    token = _event_callback_var.set(event_callback)
    try:
        final_state = pipeline.invoke(initial_state, config=config)
        _emit_event(final_state, "pipeline_complete", {
            "final_code": final_state.get("final_code"),
            "rollback_count": final_state.get("rollback_count", 0),
            "circuit_state": final_state.get("circuit_state", "CLOSED"),
            "drift_history": final_state.get("drift_history", []),
            "retry_count_planner": final_state.get("retry_count_planner", 0),
            "retry_count_coder": final_state.get("retry_count_coder", 0),
        })
    finally:
        _event_callback_var.reset(token)

    return final_state


def run_baseline_pipeline(
    task_prompt: str,
    constraints: Optional[List[str]] = None,
    fault_injection_enabled: bool = False,
    circuit_breaker_enabled: bool = False,
    thread_id: Optional[str] = None,
    checkpointer: Optional[MemorySaver] = None,
) -> AgentWorkflowState:
    """Backward-compatible runner for baseline execution."""
    return run_pipeline(
        task_prompt=task_prompt,
        constraints=constraints,
        fault_injection_enabled=fault_injection_enabled,
        circuit_breaker_enabled=circuit_breaker_enabled,
        thread_id=thread_id,
        checkpointer=checkpointer,
    )


def simulate_agent_run(
    task: str = "Implement a high-throughput binary protocol parser.",
    inject_hallucination: bool = False,
    max_iterations: int = 3,
    circuit_breaker: Optional[Any] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Backward-compatible simulation endpoint used by FastAPI and verification suites.
    """
    cb_enabled = (circuit_breaker is not None) or kwargs.get("circuit_breaker_enabled", True)
    final_state = run_pipeline(
        task_prompt=task,
        constraints=["Produce clean Python code", "Ensure high performance"],
        fault_injection_enabled=inject_hallucination,
        circuit_breaker_enabled=cb_enabled,
    )

    breaker_status = {
        "state": final_state.get("circuit_state", "CLOSED"),
        "tripped": final_state.get("circuit_state") == "OPEN",
        "rollback_count": final_state.get("rollback_count", 0),
    }

    return {
        "task": task,
        "final_code": final_state.get("final_code"),
        "review_feedback": final_state.get("review_feedback"),
        "injected_hallucination": inject_hallucination,
        "rollback_occurred": final_state.get("rollback_count", 0) > 0,
        "step_logs_count": len(final_state.get("step_logs", [])),
        "drift_history": final_state.get("drift_history", []),
        "breaker_telemetry": breaker_status,
        "active_claims": final_state.get("active_claims", []),
        "safe_checkpoints": final_state.get("safe_checkpoints", []),
    }
