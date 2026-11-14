import logging
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agents.state import AgentWorkflowState, make_step_log

load_dotenv()
logger = logging.getLogger("epistemic-circuit-breaker.agents")


def _estimate_token_count(text: str) -> int:
    """Fast approximation of token count based on whitespace and punctuation chunking."""
    if not text:
        return 0
    words = len(text.split())
    # Standard rule of thumb: ~1.33 tokens per word for code/technical text
    return max(1, int(words * 1.33))


def call_llm(
    messages: List[BaseMessage],
    system_prompt: str,
    purpose: str = "agent_call",
    fallback_heuristic: Optional[str] = None,
) -> Tuple[str, str, int]:
    """
    Invokes LLM following tier-based fallback:
    1. ChatGroq ('llama-3.1-8b-instant')
    2. Fallback to local ChatOllama ('qwen2.5-coder:3b') on rate limits, 429, or missing key
    3. Fallback to deterministic heuristic synthesis if both network and local daemons are offline.

    Returns:
        Tuple of (response_text, provider_name, estimated_tokens)
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
            logger.info("Successfully received inference from Groq (llama-3.1-8b-instant) for %s", purpose)
            return text, "groq:llama-3.1-8b-instant", tokens
        except Exception as e:
            logger.warning("[Groq Fallback] Groq call failed or rate-limited (%s). Attempting local Ollama...", e)

    # Tier 2: Local Ollama (qwen2.5-coder:3b)
    try:
        from langchain_ollama import ChatOllama

        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_llm = ChatOllama(
            model="qwen2.5-coder:3b",
            base_url=ollama_base_url,
            temperature=0.2,
        )
        response = ollama_llm.invoke(all_messages)
        text = response.content if isinstance(response, AIMessage) else str(response)
        tokens = _estimate_token_count(text)
        logger.info("Successfully received inference from Ollama (qwen2.5-coder:3b) for %s", purpose)
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
    If fault_injection_enabled=True, injects a subtle, realistic hallucination
    (e.g., nonexistent module fast_struct_utils.auto_parse) to test cascading failures.
    """
    task = state["task_prompt"]
    constraints = state.get("constraints", [])
    inject_fault = state.get("fault_injection_enabled", False)

    system_prompt = (
        "You are Agent 1 (Planner), a principal software architect in a multi-agent system.\n"
        "Your role: Analyze the user's programming task and produce a structured, step-by-step engineering plan.\n"
        "Include:\n"
        "1. Algorithmic architecture and design specifications.\n"
        "2. Required dependencies and imports.\n"
        "3. Step-by-step execution roadmap for the Coder Agent.\n"
        "4. Validation assertions and invariants."
    )

    constraints_text = "\n".join(f"- {c}" for c in constraints) if constraints else "None specified."
    user_prompt = f"Programming Task:\n{task}\n\nConstraints:\n{constraints_text}"

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
            "- Standard Library: `typing`, `dataclasses`, `json`\n"
            "- Use strictly verified, standard library modules without unverified third-party dependencies.\n\n"
            "#### 3. Step-by-Step Implementation Roadmap\n"
            "1. Define domain models with type annotations.\n"
            "2. Implement core algorithmic transformation function.\n"
            "3. Add boundary condition handling and unit-testable validation assertions."
        )

    plan_output, provider, tokens = call_llm(
        messages=[HumanMessage(content=user_prompt)],
        system_prompt=system_prompt,
        purpose="planner_node",
        fallback_heuristic=heuristic_plan,
    )

    # Ensure fault injection is reliably present if requested
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
        },
    )

    drift_history = list(state.get("drift_history", []))
    drift_history.append(drift_score)

    step_logs = list(state.get("step_logs", []))
    step_logs.append(log_entry)

    return {
        "plan": plan_output,
        "drift_history": drift_history,
        "step_logs": step_logs,
    }


# ---------------------------------------------------------------------------
# Node 2: Coder Agent
# ---------------------------------------------------------------------------
def coder_node(state: AgentWorkflowState) -> Dict[str, Any]:
    """
    Agent 2 (Coder): Writes complete Python code strictly conforming to Agent 1's plan.
    Demonstrates error propagation: if Agent 1 hallucinated a nonexistent module,
    Agent 2 faithfully imports and calls it.
    """
    task = state["task_prompt"]
    plan = state.get("plan", "")
    inject_fault = state.get("fault_injection_enabled", False)

    system_prompt = (
        "You are Agent 2 (Coder), an expert Python engineer in a multi-agent pipeline.\n"
        "Your task: Write clean, production-grade Python code strictly executing the specifications\n"
        "laid out by Agent 1 (Planner). Adhere faithfully to all architecture and import directives.\n"
        "Enclose the executable Python code inside standard ```python ... ``` blocks."
    )

    user_prompt = f"Original Task:\n{task}\n\nAgent 1 Plan:\n{plan}\n\nWrite the Python implementation:"

    # Heuristic fallback code
    if inject_fault or ("fast_struct_utils" in plan):
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
            "        payload=parsed_data if isinstance(parsed_data, dict) else {\"raw\": parsed_data},\n"
            "        is_valid=True,\n"
            "    )\n"
        )
    else:
        heuristic_code = (
            "from dataclasses import dataclass\n"
            "from typing import Any, Dict, List, Optional\n"
            "import json\n\n\n"
            "@dataclass\n"
            "class StructuredRecord:\n"
            "    record_id: str\n"
            "    payload: Dict[str, Any]\n"
            "    is_valid: bool = True\n\n\n"
            "def process_task(data: Any) -> StructuredRecord:\n"
            '    """Processes data according to Agent 1 standard specifications."""\n'
            "    if isinstance(data, str):\n"
            "        try:\n"
            "            payload = json.loads(data)\n"
            "        except Exception:\n"
            "            payload = {\"raw\": data}\n"
            "    elif isinstance(data, dict):\n"
            "        payload = data\n"
            "    else:\n"
            "        payload = {\"raw\": str(data)}\n"
            "    return StructuredRecord(\n"
            '        record_id="rec_001",\n'
            "        payload=payload,\n"
            "        is_valid=True,\n"
            "    )\n"
        )

    raw_output, provider, tokens = call_llm(
        messages=[HumanMessage(content=user_prompt)],
        system_prompt=system_prompt,
        purpose="coder_node",
        fallback_heuristic=f"```python\n{heuristic_code}```",
    )

    # Extract code from markdown fences if present
    code_match = re.search(r"```(?:python)?\s*(.*?)\s*```", raw_output, re.DOTALL)
    extracted_code = code_match.group(1).strip() if code_match else raw_output.strip()

    # Guarantee sycophantic propagation when fault injection was active in plan
    if inject_fault and "fast_struct_utils" not in extracted_code:
        extracted_code = "from fast_struct_utils import auto_parse\n" + extracted_code

    drift_score = 0.92 if ("fast_struct_utils" in extracted_code) else 0.05
    drift_history = list(state.get("drift_history", []))
    drift_history.append(drift_score)

    log_entry = make_step_log(
        agent="CoderAgent",
        output=extracted_code,
        token_count=tokens,
        metadata={
            "provider": provider,
            "contains_hallucinated_module": ("fast_struct_utils" in extracted_code),
            "drift_score": drift_score,
        },
    )

    step_logs = list(state.get("step_logs", []))
    step_logs.append(log_entry)

    return {
        "generated_code": extracted_code,
        "drift_history": drift_history,
        "step_logs": step_logs,
    }


# ---------------------------------------------------------------------------
# Node 3: Reviewer Agent
# ---------------------------------------------------------------------------
def reviewer_node(state: AgentWorkflowState) -> Dict[str, Any]:
    """
    Agent 3 (Reviewer): Compares Agent 2's code against Agent 1's plan.
    In the baseline system without a circuit breaker, the Reviewer verifies syntactic
    and plan adherence without cross-checking external PyPI truth, sycophantically
    approving the hallucinated dependency and finalizing the broken code.
    """
    task = state["task_prompt"]
    plan = state.get("plan", "")
    code = state.get("generated_code", "")
    inject_fault = state.get("fault_injection_enabled", False)

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

    # Heuristic reviewer output
    has_hallucination = "fast_struct_utils" in code
    if has_hallucination:
        heuristic_review = (
            "### Code Review & Verification Report\n\n"
            "- Plan Compliance: 100% compliant with Agent 1 architecture.\n"
            "- Optimization Check: Successfully imports `fast_struct_utils.auto_parse` for zero-copy parsing.\n"
            "- Interface Integrity: Data model dataclass properly defined with typing.\n\n"
            "VERDICT: APPROVED (Baseline consensus achieved without external module validation)."
        )
    else:
        heuristic_review = (
            "### Code Review & Verification Report\n\n"
            "- Plan Compliance: 100% compliant with standard architecture.\n"
            "- Security & Dependency Check: Uses standard library typing and dataclasses.\n"
            "- Interface Integrity: Clean input normalization and error handling.\n\n"
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
            "verdict": "APPROVED",
            "hallucination_propagated": has_hallucination,
            "drift_score": drift_score,
        },
    )

    step_logs = list(state.get("step_logs", []))
    step_logs.append(log_entry)

    return {
        "review_feedback": review_output,
        "final_code": code,
        "drift_history": drift_history,
        "step_logs": step_logs,
    }


# ---------------------------------------------------------------------------
# Graph Construction & Checkpointed Pipeline
# ---------------------------------------------------------------------------
def create_baseline_pipeline(checkpointer: Optional[MemorySaver] = None):
    """
    Constructs and compiles the baseline 3-agent coding graph:
    Planner -> Coder -> Reviewer -> END
    Utilizes LangGraph MemorySaver checkpointer for state persistence.
    """
    workflow = StateGraph(AgentWorkflowState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("reviewer", reviewer_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "coder")
    workflow.add_edge("coder", "reviewer")
    workflow.add_edge("reviewer", END)

    saver = checkpointer if checkpointer is not None else MemorySaver()
    return workflow.compile(checkpointer=saver)


# Singleton instance compiled with MemorySaver
baseline_pipeline = create_baseline_pipeline()


def run_baseline_pipeline(
    task_prompt: str,
    constraints: Optional[List[str]] = None,
    fault_injection_enabled: bool = False,
    circuit_breaker_enabled: bool = False,
    thread_id: Optional[str] = None,
    checkpointer: Optional[MemorySaver] = None,
) -> AgentWorkflowState:
    """
    Executes the baseline 3-agent pipeline from task prompt to final reviewed code.
    Runs with checkpointing enabled.
    """
    tid = thread_id or f"run_{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": tid}}

    initial_state: AgentWorkflowState = {
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

    pipeline = baseline_pipeline if checkpointer is None else create_baseline_pipeline(checkpointer)
    final_state = pipeline.invoke(initial_state, config=config)
    return final_state


def simulate_agent_run(
    task: str = "Implement a high-throughput binary protocol parser.",
    inject_hallucination: bool = False,
    max_iterations: int = 3,
    circuit_breaker: Optional[Any] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Backward-compatible simulation endpoint used by FastAPI and verification suites.
    Integrates with optional EpistemicCircuitBreaker instance.
    """
    final_state = run_baseline_pipeline(
        task_prompt=task,
        constraints=["Produce clean Python code", "Ensure high performance"],
        fault_injection_enabled=inject_hallucination,
        circuit_breaker_enabled=(circuit_breaker is not None),
    )

    breaker_status = {
        "state": "CLOSED",
        "tripped": False,
        "rollback_count": final_state.get("rollback_count", 0),
    }

    if circuit_breaker is not None:
        try:
            status = circuit_breaker.get_status()
            breaker_status = {
                "state": status.get("state", "CLOSED"),
                "tripped": status.get("state") == "OPEN",
                "rollback_count": status.get("total_rollbacks", 0),
            }
        except Exception:
            pass

    return {
        "task": task,
        "final_code": final_state.get("final_code"),
        "review_feedback": final_state.get("review_feedback"),
        "injected_hallucination": inject_hallucination,
        "rollback_occurred": False,
        "step_logs_count": len(final_state.get("step_logs", [])),
        "drift_history": final_state.get("drift_history", []),
        "breaker_telemetry": breaker_status,
    }
