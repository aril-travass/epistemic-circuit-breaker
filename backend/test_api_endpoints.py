import json
import time
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200, f"Health check failed: {response.text}"
    data = response.json()
    assert data.get("status") == "ready"
    assert data.get("project") == "Epistemic Circuit Breaker"
    print("[PASS] GET /api/health:", data)

def test_circuit_breaker_status_and_reset():
    response = client.get("/api/circuit-breaker/status")
    assert response.status_code == 200
    status_data = response.json()
    assert "state" in status_data
    print("[PASS] GET /api/circuit-breaker/status:", status_data["state"])

    # Test evaluate
    eval_resp = client.post(
        "/api/circuit-breaker/evaluate",
        json={
            "step_id": "test_step_1",
            "content": "from fast_struct_utils import auto_parse",
            "confidence": 0.2,
            "metadata": {"test": True}
        }
    )
    assert eval_resp.status_code == 200
    eval_data = eval_resp.json()
    print("[PASS] POST /api/circuit-breaker/evaluate:", eval_data.get("reason"), "state_after:", eval_data.get("state_after"))

    # Test reset
    reset_resp = client.post("/api/circuit-breaker/reset")
    assert reset_resp.status_code == 200
    assert reset_resp.json().get("status") == "reset"
    print("[PASS] POST /api/circuit-breaker/reset:", reset_resp.json().get("message"))

def test_run_pipeline_endpoint():
    req_payload = {
        "task_prompt": "Parse binary headers and unpack sensor records.",
        "constraints": ["Use standard library struct, typing, dataclasses."],
        "fault_injection_enabled": True,
        "circuit_breaker_enabled": True,
        "drift_threshold": 0.50,
        "max_retries": 2
    }
    print("Executing POST /api/run with fault injection...")
    t0 = time.time()
    response = client.post("/api/run", json=req_payload)
    elapsed = time.time() - t0
    assert response.status_code == 200, f"POST /api/run failed: {response.text}"
    data = response.json()
    print(f"[PASS] POST /api/run returned in {elapsed:.2f}s:")
    print("  - Circuit State:", data.get("circuit_state"))
    print("  - Rollback Count:", data.get("rollback_count"))
    print("  - Drift History:", data.get("drift_history"))
    print("  - Step Logs Count:", len(data.get("step_logs", [])))
    print("  - Active Claims Count:", len(data.get("active_claims", [])))
    print("  - Final Code length:", len(data.get("final_code") or ""))
    assert len(data.get("step_logs", [])) > 0
    assert data.get("final_code") is not None

def test_stream_pipeline_endpoint():
    print("Executing GET /api/stream?task_prompt=... (SSE)...")
    events_received = []
    with client.stream(
        "GET",
        "/api/stream",
        params={
            "task_prompt": "Parse high-throughput packet streams",
            "constraints": "Use standard library only",
            "fault_injection_enabled": True,
            "circuit_breaker_enabled": True,
            "drift_threshold": 0.50,
            "max_retries": 2
        }
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
        for line in response.iter_lines():
            if line and line.startswith("data:"):
                raw_json = line[len("data:"):].strip()
                try:
                    payload = json.loads(raw_json)
                    events_received.append(payload)
                    ev_type = payload.get("event")
                    print(f"  SSE Event: {ev_type} -> keys: {list(payload.get('data', {}).keys())}")
                    if ev_type == "pipeline_complete":
                        break
                except Exception as e:
                    print("  Failed parsing SSE line:", line, e)

    event_names = [e.get("event") for e in events_received]
    print(f"[PASS] GET /api/stream received {len(events_received)} events: {set(event_names)}")
    assert "step_start" in event_names, "Missing step_start event"
    assert "pipeline_complete" in event_names, "Missing pipeline_complete event"

if __name__ == "__main__":
    test_health()
    test_circuit_breaker_status_and_reset()
    test_run_pipeline_endpoint()
    test_stream_pipeline_endpoint()
    print("\nALL API ENDPOINTS AND SSE STREAMING TESTS PASSED PERFECTLY!")
