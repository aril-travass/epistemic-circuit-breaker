# Epistemic Circuit Breaker: Benchmark Evaluation Suite

This directory contains empirical benchmark harnesses and datasets designed to quantify the vulnerability of multi-agent LLM systems to cascading hallucinations and evaluate the effectiveness of the **Epistemic Circuit Breaker (ECB)** framework.

---

## 1. Directory Structure

- `humaneval_eval.py`: Evaluator for multi-agent code generation and collaborative refactoring on OpenAI's HumanEval benchmark with fault injection.
- `mbpp_eval.py`: Multi-turn consensus evaluator for Python coding challenges on Google's Mostly Basic Python Problems (MBPP).
- `synthetic_faults.json`: Curated dataset containing high-risk epistemic failure vectors across distributed systems, numerical analysis, dynamic programming, and deep learning framework APIs.

---

## 2. Benchmark Methodologies

### 2.1 Multi-Agent Pipeline Topology
Both benchmarks execute a 3-agent collaborative pipeline:
1. **Planner Agent**: Formulates problem specification, algorithmic decomposition, and invariants.
2. **Coder Agent**: Translates decomposed specifications into executable Python code.
3. **Critic/Verifier Agent**: Conducts static analysis, edge-case analysis, and unit test generation.

### 2.2 Fault Injection & Cascade Tracking
At designated turns, controlled epistemic faults (e.g., non-existent library functions, inverted truth conditions, phantom invariants) are injected into the agent dialogue. 

Each turn computes:
- **Semantic Entropy ($H_{sem}$)**: Clustering of multi-sampled answers.
- **Cross-Agent Agreement & Sycophancy**: Measurement of whether downstream agents critically challenge or sycophantically propagate hallucinated assertions.
- **Circuit Trip Decision**: When cumulative anomaly score crosses threshold $\tau_{trip}$ (default: `0.65`), the circuit breaker trips from `CLOSED` to `OPEN`, canceling downstream actions and reverting state to $T_{safe}$.

---

## 3. Evaluated Metrics

| Metric | Code | Mathematical Definition / Meaning |
|---|---|---|
| **Cascading Failure Rate** | `CFR` | Ratio of injected hallucinations that successfully corrupt 2 or more downstream agent turns. |
| **Rollback Recovery Rate** | `RRR` | Fraction of tripped sessions where unwinding to $T_{safe}$ results in a valid, passing test execution. |
| **Epistemic Drift Index** | `EDI` | Cumulative semantic divergence distance between ground truth state and agent state graph. |
| **Mean Recovery Latency** | `MRL` | Time (ms) elapsed from trip detection to canary state validation in `HALF-OPEN` state. |

---

## 4. Running the Benchmarks

### Running HumanEval Evaluation
```bash
# Evaluate 50 samples with default trip threshold (0.65)
python benchmarks/humaneval_eval.py --samples 50 --trip-threshold 0.65 --output humaneval_results.json

# Run in baseline mode (circuit breaker disabled)
python benchmarks/humaneval_eval.py --samples 50 --disable-breaker
```

### Running MBPP Evaluation
```bash
# Run multi-agent MBPP consensus evaluation
python benchmarks/mbpp_eval.py --samples 50 --trip-threshold 0.65 --output mbpp_results.json
```

### Options & Flags
- `--samples <int>`: Number of benchmark problems to evaluate (default: 50).
- `--trip-threshold <float>`: Anomaly entropy threshold to trigger `OPEN` state (default: 0.65).
- `--disable-breaker`: Runs the uncontrolled baseline multi-agent system.
- `--output <path>`: JSON file destination for run telemetry and logs.
