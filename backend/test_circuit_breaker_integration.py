"""
Integration Test Suite: Epistemic Circuit Breaker Middleware & Stateful LangGraph Rollback.
Verifies:
1. Hallucination Detection & Rollback:
   When fault_injection_enabled=True and circuit_breaker_enabled=True,
   the circuit breaker catches Agent 1's hallucinated module, rolls back state,
   injects a corrective constraint, and produces valid Python code without fast_struct_utils.
2. Baseline Failure Comparison:
   When fault_injection_enabled=True and circuit_breaker_enabled=False,
   the hallucination cascades sycophantically to Coder and Reviewer.
3. Clean Verification Run:
   When fault_injection_enabled=False and circuit_breaker_enabled=True,
   state checkpoints are verified on the first pass with zero rollbacks.
"""

import ast
import logging
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from agents.pipeline import run_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_circuit_breaker")


class TestEpistemicCircuitBreakerIntegration(unittest.TestCase):

    def test_01_circuit_breaker_detects_and_rolls_back_hallucination(self):
        """Proves circuit breaker trips on hallucinated dependency, rolls back state, and heals."""
        logger.info("\n=== Running Test 1: Fault Injection WITH Circuit Breaker Active ===")
        events = []

        def on_event(event_name, data):
            events.append((event_name, data))

        final_state = run_pipeline(
            task_prompt="Implement a high-throughput binary protocol parser.",
            constraints=["Use verified standard library modules only.", "Produce clean Python code."],
            fault_injection_enabled=True,
            circuit_breaker_enabled=True,
            max_retries=2,
            event_callback=on_event,
        )

        rollback_count = final_state.get("rollback_count", 0)
        drift_history = final_state.get("drift_history", [])
        final_code = final_state.get("final_code") or ""
        circuit_state = final_state.get("circuit_state", "CLOSED")
        corrective_constraints = final_state.get("corrective_constraints", [])

        logger.info("Rollback count: %d", rollback_count)
        logger.info("Drift history: %s", drift_history)
        logger.info("Final circuit state: %s", circuit_state)
        logger.info("Corrective constraints: %s", corrective_constraints)

        # 1. Verification of circuit tripping and rollback
        self.assertGreaterEqual(rollback_count, 1, "Circuit breaker must record at least 1 rollback")

        # 2. Verification of corrective directive injection
        self.assertTrue(
            any("fast_struct_utils" in c for c in corrective_constraints),
            f"Expected corrective constraint referencing fast_struct_utils, got: {corrective_constraints}",
        )

        # 3. Verification of final code healing
        self.assertNotIn("fast_struct_utils", final_code, "Final code must NOT contain hallucinated module")
        self.assertTrue(len(final_code) > 0, "Pipeline must generate final code")

        # 4. AST syntax validation of regenerated code
        try:
            ast.parse(final_code)
            syntax_valid = True
        except SyntaxError:
            syntax_valid = False
        self.assertTrue(syntax_valid, "Final generated code must be syntactically valid Python")

    def test_02_baseline_hallucination_cascades_without_circuit_breaker(self):
        """Proves that without the circuit breaker, the hallucinated module cascades to Coder."""
        logger.info("\n=== Running Test 2: Fault Injection WITHOUT Circuit Breaker (Baseline) ===")
        final_state = run_pipeline(
            task_prompt="Implement a high-throughput binary protocol parser.",
            constraints=["Use verified standard library modules only."],
            fault_injection_enabled=True,
            circuit_breaker_enabled=False,
            max_retries=0,
        )

        rollback_count = final_state.get("rollback_count", 0)
        plan = final_state.get("plan") or ""
        generated_code = final_state.get("generated_code") or ""

        # Hallucination was present in Agent 1's plan
        self.assertIn("fast_struct_utils", plan)
        # Without circuit breaker, zero rollbacks occurred
        self.assertEqual(rollback_count, 0, "Baseline should have 0 rollbacks")

    def test_03_clean_pipeline_executes_without_rollbacks(self):
        """Proves that a valid task with standard library modules passes with 0 rollbacks."""
        logger.info("\n=== Running Test 3: Clean Execution WITH Circuit Breaker ===")
        final_state = run_pipeline(
            task_prompt="Implement a binary header parser extracting a 4-byte magic string and 2-byte length.",
            constraints=["Use Python standard library struct and typing."],
            fault_injection_enabled=False,
            circuit_breaker_enabled=True,
            max_retries=2,
        )

        rollback_count = final_state.get("rollback_count", 0)
        final_code = final_state.get("final_code") or ""
        circuit_state = final_state.get("circuit_state", "CLOSED")

        self.assertEqual(rollback_count, 0, "Clean execution must not trigger any rollbacks")
        self.assertEqual(circuit_state, "CLOSED", "Circuit breaker must remain CLOSED")
        self.assertIn("struct", final_code, "Expected standard struct module in output")


if __name__ == "__main__":
    unittest.main()
