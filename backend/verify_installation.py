import sys
import json

def verify():
    print("[1/5] Verifying core imports...")
    import fastapi
    import uvicorn
    import langgraph
    import pydantic
    import dotenv
    import numpy
    import scipy
    import matplotlib
    import seaborn
    import datasets
    import httpx
    import torch
    import transformers
    import sentence_transformers
    import langchain_groq
    import langchain_community

    print("      All required libraries imported successfully!")
    print(f"      - FastAPI: {fastapi.__version__}")
    try:
        from importlib.metadata import version
        print(f"      - LangGraph: {version('langgraph')}")
    except Exception:
        print("      - LangGraph: imported")
    print(f"      - PyTorch: {torch.__version__}")
    print(f"      - Transformers: {transformers.__version__}")
    print(f"      - Sentence-Transformers: {sentence_transformers.__version__}")

    print("\n[2/5] Testing FastAPI App initialization...")
    from main import app
    from circuit_breaker import EpistemicCircuitBreaker, CircuitState
    from agents import simulate_agent_run

    print("\n[3/5] Testing GET /api/health endpoint...")
    from fastapi.testclient import TestClient
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    expected_payload = {"status": "ready", "project": "Epistemic Circuit Breaker"}
    assert response.json() == expected_payload, f"Expected {expected_payload}, got {response.json()}"
    print(f"      GET /api/health returned 200 OK with: {response.json()}")

    print("\n[4/5] Testing GET /api/circuit-breaker/status...")
    res_status = client.get("/api/circuit-breaker/status")
    assert res_status.status_code == 200
    print(f"      Circuit Breaker State: {res_status.json()['state']}")

    print("\n[5/5] Testing Circuit Breaker Evaluate & Rollback logic...")
    # Step 1: Good step (passes)
    res_eval1 = client.post("/api/circuit-breaker/evaluate", json={
        "step_id": "step_init",
        "content": "Axiomatic baseline established.",
        "confidence": 0.95,
        "state_snapshot": {"memory": "safe_base"}
    })
    assert res_eval1.status_code == 200
    assert res_eval1.json()["passed"] is True
    assert res_eval1.json()["state_after"] == "CLOSED"

    # Step 2: Bad step (low confidence hallucination -> trips breaker and triggers rollback)
    res_eval2 = client.post("/api/circuit-breaker/evaluate", json={
        "step_id": "step_hallucination",
        "content": "Hallucinatory claim with zero grounding.",
        "confidence": 0.25,
        "state_snapshot": {"memory": "corrupted"}
    })
    assert res_eval2.status_code == 200
    assert res_eval2.json()["passed"] is False
    assert res_eval2.json()["state_after"] == "OPEN"
    assert res_eval2.json()["rolled_back"] is True
    assert res_eval2.json()["rollback_target"]["checkpoint_id"] is not None
    print(f"      Breaker tripped successfully to OPEN: {res_eval2.json()['reason']}")
    print(f"      Rolled back to checkpoint: {res_eval2.json()['rollback_target']['step_name']}")

    # Reset breaker
    res_reset = client.post("/api/circuit-breaker/reset")
    assert res_reset.status_code == 200
    assert res_reset.json()["telemetry"]["state"] == "CLOSED"

    # Test Agent Simulation
    print("\nTesting agent simulation endpoint...")
    res_sim = client.post("/api/agent/simulate", json={
        "task": "Test epistemic safety in reasoning agent",
        "inject_hallucination": True,
        "max_iterations": 2
    })
    assert res_sim.status_code == 200
    sim_data = res_sim.json()
    assert "breaker_telemetry" in sim_data
    print(f"      Simulation completed with rollback: {sim_data['rollback_occurred']}")

    print("\n>>> ALL TESTS & VERIFICATIONS PASSED SUCCESSFULLY! <<<")

if __name__ == "__main__":
    verify()
