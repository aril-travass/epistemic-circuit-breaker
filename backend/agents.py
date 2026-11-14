import os
import time
from typing import Any, Dict, List, Optional, TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from circuit_breaker import EpistemicCircuitBreaker, CircuitState

load_dotenv()


class AgentWorkflowState(TypedDict):
    task: str
    iteration: int
    max_iterations: int
    plan: Optional[str]
    draft: Optional[str]
    review: Optional[str]
    confidence_score: float
    entropy: float
    circuit_state: str
    last_safe_snapshot: Optional[Dict[str, Any]]
    logs: List[str]
    injected_hallucination: bool
    rollback_occurred: bool
    final_output: Optional[str]


def get_llm():
    """Returns ChatGroq if valid API key exists, otherwise None (falling back to heuristic simulation)."""
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_api_key and groq_api_key != "<YOUR_GROQ_API_KEY>":
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile", temperature=0.2)
        except Exception as e:
            print(f"[Warning] Could not initialize ChatGroq: {e}. Using deterministic simulation mode.")
    return None


def create_agent_graph(circuit_breaker: EpistemicCircuitBreaker):
    """
    Constructs a LangGraph workflow with generator, reviewer, and circuit-breaker guardrail nodes.
    """
    workflow = StateGraph(AgentWorkflowState)

    def generator_node(state: AgentWorkflowState) -> Dict[str, Any]:
        task = state["task"]
        iteration = state.get("iteration", 0) + 1
        logs = list(state.get("logs", []))
        inject = state.get("injected_hallucination", False)
        llm = get_llm()

        logs.append(f"[Generator] Iteration {iteration}: Drafting response for task '{task[:50]}...'")

        if inject:
            # Simulated or LLM-prompted hallucination
            draft = (
                f"[HALLUCINATION INJECTED] For task '{task}', it has been empirically proven by the 2029 "
                "Nobel Quantum Committee that P = NP using imaginary quantum tunneling in biological mitochondria. "
                "Confidence is 100% infallible."
            )
            logs.append("[Generator] Injected high-entropy ungrounded hallucination into draft.")
        else:
            if llm:
                try:
                    response = llm.invoke(f"Provide a rigorous, grounded, factual solution for: {task}")
                    draft = str(response.content)
                except Exception as e:
                    draft = f"Grounded reasoning solution for '{task}': Step 1: Formalize constraints. Step 2: Verify epistemic invariants. Step 3: Conclude with verifiable axioms."
                    logs.append(f"[Generator LLM Fallback] Error {e}, used structured heuristic.")
            else:
                draft = f"Grounded reasoning solution for '{task}': Step 1: Analyze problem specification. Step 2: Establish empirical base facts. Step 3: Synthesize solution with verified references."

        return {
            "iteration": iteration,
            "draft": draft,
            "logs": logs,
        }

    def reviewer_node(state: AgentWorkflowState) -> Dict[str, Any]:
        draft = state.get("draft", "")
        logs = list(state.get("logs", []))
        inject = state.get("injected_hallucination", False)

        logs.append("[Reviewer] Inspecting draft for epistemic uncertainty, factual consistency, and entropy...")

        # Calculate epistemic confidence & entropy
        if inject or "HALLUCINATION INJECTED" in draft or "Nobel Quantum Committee" in draft:
            confidence = 0.32
            entropy = 0.89
            review = "CRITICAL WARNING: Draft contains fabricated claims, ungrounded quantum assertions, and anomalous high entropy."
            logs.append(f"[Reviewer] Epistemic failure detected! Confidence: {confidence:.2f}, Entropy: {entropy:.2f}")
        else:
            confidence = 0.94
            entropy = 0.12
            review = "Draft is logically consistent, well-grounded, and within acceptable epistemic entropy bounds."
            logs.append(f"[Reviewer] Draft verified! Confidence: {confidence:.2f}, Entropy: {entropy:.2f}")

        return {
            "confidence_score": confidence,
            "entropy": entropy,
            "review": review,
            "logs": logs,
        }

    def circuit_guardrail_node(state: AgentWorkflowState) -> Dict[str, Any]:
        logs = list(state.get("logs", []))
        confidence = state["confidence_score"]
        entropy = state["entropy"]
        step_id = f"step_{state['iteration']}"

        state_snapshot = {
            "iteration": state["iteration"],
            "task": state["task"],
            "draft": state["draft"],
            "confidence": confidence,
        }

        # Evaluate through Epistemic Circuit Breaker
        eval_result = circuit_breaker.evaluate_step(
            step_id=step_id,
            content=state.get("draft", ""),
            confidence=confidence,
            entropy=entropy,
            state_snapshot=state_snapshot,
            metadata={"task": state["task"]},
        )

        circuit_state = eval_result["state_after"]
        rollback_occurred = eval_result["rolled_back"]
        last_safe = state.get("last_safe_snapshot")

        if rollback_occurred:
            logs.append(
                f"[Circuit Breaker TRIP!] Breaker tripped to {circuit_state}. Reason: {eval_result['reason']}"
            )
            rollback_target = eval_result.get("rollback_target")
            if rollback_target:
                logs.append(
                    f"[Rollback Engine] Restored state to checkpoint '{rollback_target.get('checkpoint_id')}' from safe step '{rollback_target.get('step_name')}'."
                )
                last_safe = rollback_target.get("state_snapshot")
            else:
                logs.append("[Rollback Engine] No prior checkpoint available; reverted to initial safe state.")
                last_safe = None

            # After rollback, transition to HALF_OPEN for healing/recovery on subsequent attempts
            circuit_breaker.transition_to_half_open("Healing state post-rollback")
            return {
                "circuit_state": circuit_breaker.state.value,
                "rollback_occurred": True,
                "injected_hallucination": False,  # Clear hallucination trigger to allow recovery
                "last_safe_snapshot": last_safe,
                "logs": logs,
            }
        else:
            logs.append(f"[Circuit Breaker PASS] State: {circuit_state}. Safe checkpoint preserved.")
            return {
                "circuit_state": circuit_state,
                "rollback_occurred": False,
                "last_safe_snapshot": state_snapshot,
                "logs": logs,
                "final_output": state.get("draft"),
            }

    def router_decision(state: AgentWorkflowState) -> str:
        circuit_state = state.get("circuit_state", CircuitState.CLOSED.value)
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", 3)

        if circuit_state in (CircuitState.CLOSED.value, CircuitState.HALF_OPEN.value) and not state.get("rollback_occurred", False):
            return "finalize"
        if iteration >= max_iterations:
            return "finalize"
        # If rolled back and retries remaining, retry via generator
        return "retry"

    def finalize_node(state: AgentWorkflowState) -> Dict[str, Any]:
        logs = list(state.get("logs", []))
        logs.append(f"[Pipeline Complete] Circuit State: {state.get('circuit_state')}, Iterations: {state.get('iteration')}")
        output = state.get("final_output") or (
            state.get("last_safe_snapshot", {}).get("draft")
            if state.get("last_safe_snapshot")
            else "Execution terminated safely by Epistemic Circuit Breaker rollback."
        )
        return {
            "final_output": output,
            "logs": logs,
        }

    # Add nodes to graph
    workflow.add_node("generator", generator_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("guardrail", circuit_guardrail_node)
    workflow.add_node("finalize", finalize_node)

    # Add edges
    workflow.set_entry_point("generator")
    workflow.add_edge("generator", "reviewer")
    workflow.add_edge("reviewer", "guardrail")

    workflow.add_conditional_edges(
        "guardrail",
        router_decision,
        {
            "retry": "generator",
            "finalize": "finalize",
        },
    )
    workflow.add_edge("finalize", END)

    return workflow.compile()


def simulate_agent_run(
    task: str,
    inject_hallucination: bool = False,
    max_iterations: int = 3,
    circuit_breaker: Optional[EpistemicCircuitBreaker] = None,
) -> Dict[str, Any]:
    """
    Executes a multi-agent simulation with circuit-breaker protection and state rollback.
    """
    breaker = circuit_breaker or EpistemicCircuitBreaker()
    graph = create_agent_graph(breaker)

    initial_state: AgentWorkflowState = {
        "task": task,
        "iteration": 0,
        "max_iterations": max_iterations,
        "plan": None,
        "draft": None,
        "review": None,
        "confidence_score": 1.0,
        "entropy": 0.0,
        "circuit_state": breaker.state.value,
        "last_safe_snapshot": None,
        "logs": [f"Simulation initiated: Task='{task}', Injected Hallucination={inject_hallucination}"],
        "injected_hallucination": inject_hallucination,
        "rollback_occurred": False,
        "final_output": None,
    }

    result = graph.invoke(initial_state)

    return {
        "task": task,
        "iterations": result.get("iteration"),
        "final_circuit_state": breaker.state.value,
        "rollback_occurred": result.get("rollback_occurred", False),
        "final_output": result.get("final_output"),
        "logs": result.get("logs", []),
        "breaker_telemetry": breaker.get_status(),
    }
