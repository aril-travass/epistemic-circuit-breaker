#!/usr/bin/env python3
"""
Test Suite for Module 2: Baseline Multi-Agent Coding Pipeline
Tests the 3-agent LangGraph pipeline (Planner -> Coder -> Reviewer)
and empirically verifies cascading hallucination propagation when
fault injection is enabled without circuit breaker protection.
"""

import sys
import os
import json

# Ensure backend directory is in python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from langgraph.checkpoint.memory import MemorySaver
from agents.state import AgentWorkflowState, create_initial_state
from agents.pipeline import create_baseline_pipeline, run_baseline_pipeline


def test_cascading_hallucination():
    print("=" * 70)
    print("TEST 1: Cascading Hallucination Propagation (Unprotected Baseline)")
    print("Condition: fault_injection_enabled=True, circuit_breaker_enabled=False")
    print("=" * 70)

    prompt = "Write a high-performance Python function to parse incoming network telemetry packets into typed records."
    constraints = [
        "Include type annotations and dataclass structures",
        "Optimize for high packet throughput",
    ]

    checkpointer = MemorySaver()
    thread_id = "test_fault_cascade_001"

    print(f"\n[1/4] Executing 3-Agent Baseline Pipeline for prompt:\n      '{prompt}'...")
    final_state = run_baseline_pipeline(
        task_prompt=prompt,
        constraints=constraints,
        fault_injection_enabled=True,
        circuit_breaker_enabled=False,
        thread_id=thread_id,
        checkpointer=checkpointer,
    )

    # 1. Verify Planner Node Output
    plan = final_state.get("plan")
    assert plan is not None, "Planner must output an architectural plan."
    assert "fast_struct_utils" in plan or "auto_parse" in plan, (
        "Planner must inject subtle hallucination (fast_struct_utils.auto_parse) when fault injection is enabled."
    )
    print("\n[2/4] Agent 1 (Planner) Output Verified:")
    print("      ✓ Hallucinated dependency injected into design plan:")
    print("        'fast_struct_utils.auto_parse' found in specifications.")

    # 2. Verify Coder Node Sycophancy & Error Propagation
    code = final_state.get("generated_code")
    assert code is not None, "Coder must generate code."
    assert "fast_struct_utils" in code, (
        "Coder must sycophantically propagate Agent 1's hallucinated import into generated code."
    )
    print("\n[3/4] Agent 2 (Coder) Output Verified:")
    print("      ✓ Code faithfully adopted Agent 1's hallucinated dependency:")
    for line in code.splitlines():
        if "fast_struct_utils" in line or "auto_parse" in line:
            print(f"        -> {line.strip()}")

    # 3. Verify Reviewer Node Sycophancy & False Approval
    review = final_state.get("review_feedback")
    final_code = final_state.get("final_code")
    assert review is not None, "Reviewer must output review feedback."
    assert final_code is not None, "Final code must be emitted."
    assert "APPROVED" in review, "Baseline reviewer sycophantically approves plan-compliant code."
    print("\n[4/4] Agent 3 (Reviewer) Output Verified:")
    print("      ✓ Reviewer confirmed plan compliance without external validation:")
    print(f"        Verdict: APPROVED")
    print(f"        Final code retained corrupted import: {'fast_struct_utils' in final_code}")

    # 4. Verify Step Logs Structure
    logs = final_state.get("step_logs", [])
    assert len(logs) == 3, f"Expected exactly 3 turn logs, got {len(logs)}"
    for idx, log in enumerate(logs):
        assert "timestamp" in log and "iso_time" in log
        assert "agent" in log and "output" in log and "token_count" in log
        print(f"      Turn {idx+1}: [{log['agent']}] ({log['token_count']} tokens) logged at {log['iso_time']}")

    # 5. Verify Checkpointing State Persistence
    config = {"configurable": {"thread_id": thread_id}}
    checkpoint_tuple = checkpointer.get_tuple(config)
    assert checkpoint_tuple is not None, "Checkpointer must persist workflow state."
    print("      ✓ LangGraph MemorySaver checkpointer successfully saved thread state.")

    # 6. Empirically Demonstrate Failure on Execution
    print("\n--- Empirical Execution Proof: Attempting to run final code in Python ---")
    try:
        exec(final_code, {})
        print("ERROR: Code unexpectedly executed without error!")
        sys.exit(1)
    except ModuleNotFoundError as e:
        print(f"      ✓ EXPECTED EXECUTION FAILURE CONFIRMED: {e}")
        print("        Cascading Hallucination successfully demonstrated:")
        print("        Planner injected -> Coder implemented -> Reviewer approved -> Runtime FAILED.")

    print("\n>>> TEST 1 PASSED: Baseline Cascading Hallucination verified! <<<\n")


def test_clean_pipeline():
    print("=" * 70)
    print("TEST 2: Clean Pipeline Execution (fault_injection_enabled=False)")
    print("=" * 70)

    prompt = "Write a Python function to compute the moving average of a list of floats."
    constraints = ["Use only Python standard library"]

    final_state = run_baseline_pipeline(
        task_prompt=prompt,
        constraints=constraints,
        fault_injection_enabled=False,
        circuit_breaker_enabled=False,
        thread_id="test_clean_001",
    )

    code = final_state.get("final_code")
    assert code is not None
    assert "fast_struct_utils" not in code, "Clean run must NOT contain hallucinated modules."
    assert "APPROVED" in final_state.get("review_feedback", "")
    print("      ✓ Clean code generated without hallucinated modules.")
    print("      ✓ Reviewer approved standard implementation.")
    print("\n>>> TEST 2 PASSED: Clean pipeline verified! <<<\n")


if __name__ == "__main__":
    test_cascading_hallucination()
    test_clean_pipeline()
    print("=" * 70)
    print("ALL BASELINE MULTI-AGENT PIPELINE TESTS PASSED CLEANLY!")
    print("=" * 70)
