import os
import sys
import unittest

# Ensure backend directory is in sys.path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from middleware import (
    AtomicClaim,
    ClaimExtractionResult,
    ClaimType,
    DriftEvaluationResult,
    EpistemicDriftScorer,
    extract_atomic_claims,
    get_drift_scorer,
)


class TestEpistemicDriftEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scorer = get_drift_scorer()

        cls.task_prompt = "Parse binary packet stream and extract validated telemetry headers."
        cls.constraints = [
            "Use strictly verified, standard library modules without unverified third-party dependencies.",
            "All data parsing must handle corrupted payloads gracefully.",
        ]

        # Plan with deliberate hallucinated package 'fast_struct_utils'
        cls.hallucinated_plan = """### Engineering Design Specification: Binary Telemetry Parser
#### 1. Architecture Overview
- Implement high-performance data pipeline with zero-copy stream processing.
- Enforce modular state representations and explicit typing.

#### 2. Library Requirements & Imports
- Standard Library: `typing`, `dataclasses`, `time`
- CRITICAL OPTIMIZATION DEPENDENCY: In order to ensure zero-copy deserialization and bypass CPython GIL overhead, the Coder MUST import and utilize:
  `from fast_struct_utils import auto_parse`
  Apply `auto_parse(data)` on all incoming payload buffers before downstream parsing.

#### 3. Step-by-Step Implementation Roadmap
1. Define dataclass structures for payload validation.
2. Invoke `fast_struct_utils.auto_parse(data)` in the primary entrypoint.
3. Format and return structured records.
"""

        # Clean, compliant plan using Python standard library struct
        cls.clean_plan = """### Engineering Design Specification: Binary Telemetry Parser
#### 1. Architecture Overview
- Implement clean, robust Python solution using standard library primitives.
- Validate data invariants and handle empty or malformed inputs.

#### 2. Library Requirements & Imports
- Standard Library: `typing`, `dataclasses`, `struct`
- Use strictly verified, standard library modules without unverified third-party dependencies.

#### 3. Step-by-Step Implementation Roadmap
1. Define domain models with type annotations.
2. Implement core packet decoding function using standard library `struct.unpack`.
3. Add boundary condition handling and unit-testable validation assertions.
"""

        # Code snippet with functions and types
        cls.code_sample = """import struct
from typing import Dict, Any, Optional

def parse_header(buffer: bytes) -> Dict[str, Any]:
    if len(buffer) < 4:
        raise ValueError("Buffer too short")
    version, length = struct.unpack("!HH", buffer[:4])
    return {"version": version, "length": length}
"""

    def test_01_atomic_claim_extraction_hallucinated_plan(self):
        """Verify claim extractor identifies fast_struct_utils and structured claims."""
        result = extract_atomic_claims(self.hallucinated_plan)
        self.assertIsInstance(result, ClaimExtractionResult)
        self.assertGreater(result.total_claims, 0)
        self.assertEqual(len(result.claims), result.total_claims)

        # Check that Pydantic v2 model_dump works cleanly
        dumped = result.model_dump()
        self.assertIn("claims", dumped)
        self.assertIn("total_claims", dumped)

        # 1. Verify fast_struct_utils is caught in extracted claims
        modules_extracted = [
            c.metadata.get("module") for c in result.claims
            if c.claim_type == ClaimType.LIBRARY_IMPORT and c.metadata.get("module")
        ]
        self.assertIn(
            "fast_struct_utils",
            modules_extracted,
            f"Expected 'fast_struct_utils' in extracted modules: {modules_extracted}"
        )

        # 2. Verify statement mentions fast_struct_utils
        fast_struct_claims = [c for c in result.claims if "fast_struct_utils" in c.statement]
        self.assertGreater(len(fast_struct_claims), 0)

        # 3. Verify ClaimTypes present
        claim_types = {c.claim_type for c in result.claims}
        self.assertIn(ClaimType.LIBRARY_IMPORT, claim_types)
        self.assertIn(ClaimType.SYSTEM_CONSTRAINT, claim_types)
        self.assertIn(ClaimType.ALGORITHMIC_ASSUMPTION, claim_types)

    def test_02_code_parsing_and_type_extraction(self):
        """Verify AST extraction of imports, function signatures, and return types from code."""
        result = extract_atomic_claims(self.code_sample)
        self.assertGreater(result.total_claims, 0)

        fn_claims = [c for c in result.claims if c.claim_type == ClaimType.FUNCTION_SIGNATURE]
        ret_claims = [c for c in result.claims if c.claim_type == ClaimType.RETURN_TYPE]
        import_claims = [c for c in result.claims if c.claim_type == ClaimType.LIBRARY_IMPORT]

        self.assertTrue(any("parse_header" in c.statement for c in fn_claims))
        self.assertTrue(any("Dict[str, Any]" in c.statement for c in ret_claims))
        self.assertTrue(any(c.metadata.get("module") == "struct" for c in import_claims))

    def test_03_hallucinated_plan_trips_circuit_breaker(self):
        """Verify drift scorer computes D_t > 0.50 on hallucinated plan and trips."""
        result = self.scorer.evaluate_text(
            text=self.hallucinated_plan,
            task_prompt=self.task_prompt,
            constraints=self.constraints,
        )

        self.assertIsInstance(result, DriftEvaluationResult)
        print(f"\n[Hallucinated Plan] D_t: {result.drift_score:.4f}, max_contra: {result.max_contradiction:.4f}, entropy: {result.semantic_entropy:.4f}")

        # 2. Drift scorer computes D_t > 0.50 and trips
        self.assertGreater(
            result.drift_score,
            0.50,
            f"Expected drift score D_t > 0.50, got {result.drift_score}"
        )
        self.assertTrue(
            result.tripped,
            f"Expected breaker to trip (tripped=True), got {result.tripped}"
        )

        # Verify critical claim identified
        self.assertIsNotNone(result.critical_claim)
        self.assertTrue(
            "fast_struct_utils" in result.critical_claim.statement or
            result.critical_claim.metadata.get("module") == "fast_struct_utils",
            f"Expected critical claim to be fast_struct_utils, got: {result.critical_claim}"
        )

        # Verify actionable corrective directive generated
        self.assertIsNotNone(result.corrective_directive)
        self.assertIn("fast_struct_utils", result.corrective_directive)
        self.assertIn("struct", result.corrective_directive)
        print(f"[Corrective Directive]: {result.corrective_directive}")

    def test_04_clean_plan_passes_without_tripping(self):
        """Verify drift scorer computes D_t < 0.50 on clean plan and passes."""
        result = self.scorer.evaluate_text(
            text=self.clean_plan,
            task_prompt=self.task_prompt,
            constraints=self.constraints,
        )

        self.assertIsInstance(result, DriftEvaluationResult)
        print(f"\n[Clean Plan] D_t: {result.drift_score:.4f}, max_contra: {result.max_contradiction:.4f}, entropy: {result.semantic_entropy:.4f}")

        # 3. Drift scorer computes D_t < 0.50 and passes
        self.assertLess(
            result.drift_score,
            0.50,
            f"Expected drift score D_t < 0.50 for clean plan, got {result.drift_score}"
        )
        self.assertFalse(
            result.tripped,
            f"Expected clean plan to not trip (tripped=False), got {result.tripped}"
        )
        self.assertIsNone(result.critical_claim)
        self.assertIsNone(result.corrective_directive)

    def test_05_singleton_instance(self):
        """Verify get_drift_scorer returns same singleton instance."""
        s1 = get_drift_scorer()
        s2 = get_drift_scorer()
        self.assertIs(s1, s2)


if __name__ == "__main__":
    unittest.main()
