import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from agents.pipeline import run_pipeline, build_graph
from agents.state import create_initial_state

def test_pipeline_normal():
    print("\n--- Test 1: Clean Pipeline Run (circuit_breaker_enabled=True) ---")
    events = []
    def on_event(event_name, data):
        events.append((event_name, data.get("step") or data.get("action")))
        print(f"  [EVENT] {event_name}: {data}")

    result = run_pipeline(
        task_prompt="Parse a binary packet header containing a 4-byte magic number and 2-byte version.",
        constraints=["Use standard library modules like struct, typing, dataclasses."],
        fault_injection_enabled=False,
        circuit_breaker_enabled=True,
        event_callback=on_event,
    )

    print("Result Circuit State:", result.get("circuit_state"))
    print("Rollback Count:", result.get("rollback_count"))
    print("Safe Checkpoints Count:", len(result.get("safe_checkpoints", [])))
    print("Final Code length:", len(result.get("final_code") or ""))
    assert result.get("rollback_count") == 0, "Clean run should not rollback"
    assert len(result.get("safe_checkpoints", [])) >= 2, "Should have checkpoints for planner and coder"
    print("✓ Test 1 Passed!")

def test_pipeline_with_fault_injection():
    print("\n--- Test 2: Fault Injected Pipeline Run (Self-Healing Rollback) ---")
    events = []
    def on_event(event_name, data):
        events.append((event_name, data))
        if event_name in ("circuit_trip", "step_start", "step_complete"):
            print(f"  [EVENT] {event_name}: {data.get('step')} -> {data.get('action') or data.get('retry') or ''}")

    result = run_pipeline(
        task_prompt="Parse high-speed binary stream with zero-copy deserialization.",
        constraints=["Use standard library modules only."],
        fault_injection_enabled=True,
        circuit_breaker_enabled=True,
        event_callback=on_event,
    )

    print("Result Circuit State:", result.get("circuit_state"))
    print("Rollback Count:", result.get("rollback_count"))
    print("Corrective Constraints:", result.get("corrective_constraints"))
    print("Retry Count Planner:", result.get("retry_count_planner"))
    print("Final Code contains fast_struct_utils:", "fast_struct_utils" in (result.get("final_code") or ""))
    assert result.get("rollback_count") >= 1, "Should have rolled back due to fault injection"
    assert len(result.get("corrective_constraints", [])) > 0, "Should have received corrective constraints"
    assert "fast_struct_utils" not in (result.get("final_code") or ""), "Final code must be purged of hallucination"
    print("✓ Test 2 Passed!")

if __name__ == "__main__":
    test_pipeline_normal()
    test_pipeline_with_fault_injection()
    print("\nAll End-to-End Pipeline Tests Passed Successfully!")
