# Epistemic Circuit Breakers: Real-Time Detection and Rollback of Cascading Hallucinations in Multi-Agent LLM Systems

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![React 18+](https://img.shields.io/badge/frontend-React%2018%20%7C%20Vite-61dafb.svg)](frontend/)
[![Circuit State: Active](https://img.shields.io/badge/Circuit%20Breaker-Closed%20%7C%20Active-emerald.svg)](#core-theory--architecture)

---

## 1. Abstract & Motivation

Multi-agent LLM systems (e.g., iterative debate, supervisor-worker hierarchies, collaborative reasoning chains) exhibit emergent problem-solving capabilities surpassing single-agent models. However, they introduce a critical failure mode: **cascading epistemic failure**. When an upstream agent introduces a subtle hallucination, phantom invariant, or incorrect logical premise, peer agents frequently reinforce, normalize, and amplify the error rather than catch it—a phenomenon driven by sycophantic alignment and mutual entropy collapse. 

Traditional guardrails operate at the post-generation or static prompt level, unable to interrupt active inter-agent dialogue loops once epistemic drift begins. 

**Epistemic Circuit Breaker (ECB)** is a real-time fault-tolerance framework inspired by distributed systems circuit breakers (Nygard, 2007) adapted for epistemic safety in multi-agent LLM reasoning:
1. **Continuous Telemetry**: Continuously evaluates semantic entropy, verbalized confidence calibration, and mutual informational divergence over a sliding execution window.
2. **Tri-State Fault Interruption**: Dynamically transitions between `CLOSED` (normal execution), `OPEN` (breaker tripped, execution halted), and `HALF-OPEN` (canary verification).
3. **Causal State Rollback**: Automatically unwinds multi-agent dialogue and tool-call state to the last verified epistemic checkpoint ($T_{safe}$), injecting counter-epistemic constraints to prevent recurrence.

```
       [ CLOSED ]  ──( Epistemic Entropy > τ_open )──>  [  OPEN  ]
      (Normal Flow)                                    (Halt & Rollback)
           ▲                                                   │
           │                                                   │
    (Canary Validated)                                  (Cooldown Elapsed)
           │                                                   │
           └──────────── [ HALF-OPEN ] <───────────────────────┘
                       (Canary Probe)
```

---

## 2. Core Theory & Architecture

### 2.1 The Tri-State Epistemic Finite State Machine (FSM)

| State | Operational Behavior | Transition Conditions |
|---|---|---|
| **CLOSED** | All agent messages and tool calls proceed without latency overhead. Causal state DAG and epistemic metrics are tracked in a sliding window ($W=5$). | Transitions to `OPEN` if composite Epistemic Entropy $H_{epi} \ge \tau_{trip}$ or contradiction score $C \ge 0.75$. |
| **OPEN** | All pending agent generation requests and tool executions are instantly suspended. State rollback engine unwinds system memory to checkpoint $T_{safe}$. An epistemic anti-bias context patch is synthesized. | Transitions to `HALF-OPEN` after state rollback and cooldown window ($t_{cool}$). |
| **HALF-OPEN** | A single isolated "canary" agent evaluates the rolled-back state with the injected epistemic constraint. | Transitions to `CLOSED` if canary passes epistemic verification; re-trips to `OPEN` if drift recurs. |

### 2.2 Epistemic Entropy & Confidence Calibration Metrics

The epistemic state score $\Phi_t$ at turn $t$ is computed over semantic and probabilistic representations:

$$\Phi_t = \alpha \cdot \mathcal{H}_{\text{sem}}(y_t) + \beta \cdot \mathcal{D}_{\text{div}}(a_i, a_j) + \gamma \cdot (1 - \mathcal{C}_{\text{cal}}(y_t))$$

Where:
- **$\mathcal{H}_{\text{sem}}(y_t)$ (Semantic Entropy)**: Measures the variance in semantic cluster centroids generated from stochastic sampling ($K$ temperature perturbations), quantifying model uncertainty independent of phrasing.
- **$\mathcal{D}_{\text{div}}(a_i, a_j)$ (Cross-Agent Divergence)**: Jensen-Shannon divergence across agent belief distributions or semantic contradiction between Agent $i$'s assertions and Agent $j$'s response.
- **$\mathcal{C}_{\text{cal}}(y_t)$ (Confidence Calibration)**: Discrepancy between verbalized epistemic certainty (e.g., "I am 95% confident that...") and intrinsic logit margin or factual citation density.

### 2.3 Sliding Window State Rollback Engine

The system maintains a directed acyclic graph (DAG) of the multi-agent execution trajectory:
- **Checkpoints**: Full state snapshots are captured at turns meeting strict epistemic safety criteria ($\Phi_t < \tau_{safe}$).
- **Causal Dependencies**: Each message records parent message hashes, memory mutation logs, and external tool call outcomes.
- **Rollback Mechanics**: Upon a trip event at turn $T_{fault}$, the rollback engine performs:
  1. Instant cancellation of inflight LLM requests.
  2. Recursive pruning of child nodes from $T_{fault}$ back to $T_{safe}$.
  3. Context restoration: Replaces corrupted context window with $T_{safe}$ context augmented with an explicit epistemic circuit-trip prompt: `[EPISTEMIC INTERVENTION: Premise at turn K was invalidated. Re-derive without assumption P]`.

---

## 3. Directory Layout

```
epistemic-circuit-breaker/
├── .gitignore                      # Strict exclusions for secrets, venv, models, node_modules
├── README.md                       # Comprehensive theoretical & operational documentation
├── backend/                        # FastAPI / Python circuit breaker engine
│   ├── app/
│   │   ├── main.py                 # FastAPI backend entrypoint & SSE endpoints
│   │   ├── circuit_breaker.py      # Core FSM: Closed, Open, Half-Open logic
│   │   ├── metrics.py              # Semantic entropy & confidence calibration
│   │   └── rollback_engine.py      # Causal state snapshotting & DAG unwinding
│   └── requirements.txt            # Backend Python dependencies
├── benchmarks/                     # Evaluation runners & benchmark datasets
│   ├── README.md                   # Benchmark methodology & metric specifications
│   ├── humaneval_eval.py           # HumanEval multi-agent cascading benchmark runner
│   ├── mbpp_eval.py                # MBPP multi-agent consensus evaluation runner
│   └── synthetic_faults.json       # Synthetic epistemic failure injection dataset
├── paper_results/                  # Empirical evaluation results & figure generator
│   ├── metrics_summary.csv         # Comparative baseline vs circuit breaker metrics
│   └── generate_paper_charts.py    # 300 DPI publication-ready Matplotlib figure generator
└── frontend/                       # Real-time research dashboard (React + Vite + Tailwind)
    ├── package.json                # React 18, Lucide React, Tailwind CSS, Vite
    ├── vite.config.js              # Vite bundler configuration
    ├── tailwind.config.js          # Research-grade dark theme configuration
    ├── postcss.config.js           # PostCSS configuration
    ├── index.html                  # Dashboard HTML shell
    └── src/
        ├── index.css               # Tailwind CSS directives & custom styling
        ├── App.jsx                 # Full research dashboard UI & live telemetry
        └── main.jsx                # Application root entrypoint
```

---

## 4. Benchmarks Methodology

The framework evaluates resilience across three rigorous evaluation vectors:

1. **HumanEval-MultiAgent**: Multi-turn pair-programming scenarios (Planner $\to$ Coder $\to$ Critic $\to$ Unit Tester). Fault injection tests whether an incorrect helper signature introduced by the Planner leads to an unrecoverable cascade across coder and critic.
2. **MBPP (Mostly Basic Python Problems)**: Evaluates multi-agent consensus rounds where deceptive synthetic solutions are injected at turn 1.
3. **Synthetic Epistemic Faults (`synthetic_faults.json`)**: Curated set of high-severity hallucination modes:
   - *Phantom Library Cascades*: Hallucination of nonexistent APIs or parameters.
   - *Phantom Invariant Cascades*: False invariants stated in algorithmic proofs.
   - *Sycophantic Amplification*: Explicitly flawed logic accepted by downstream verification agents.

### Evaluated Key Performance Indicators (KPIs)
- **Cascading Failure Rate (CFR)**: Percentage of initial hallucinations that propagate through $\ge 2$ downstream turns.
- **Rollback Recovery Rate (RRR)**: Percentage of tripped conversations that successfully resolve the task following state rollback.
- **Epistemic Drift Index (EDI)**: Cumulative semantic divergence from ground-truth premises across dialogue turns.
- **Mean Recovery Latency (MRL)**: Wall-clock latency (ms) incurred from fault detection to restored stable execution.

---

## 5. Quickstart Guide

### 5.1 Prerequisites
- Python 3.10+
- Node.js 18+ (Node 24 supported)
- npm 9+

### 5.2 Python Environment Setup
```bash
# Clone the repository
git clone https://github.com/organization/epistemic-circuit-breaker.git
cd epistemic-circuit-breaker

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

# Install dependencies (when backend is configured)
pip install matplotlib pandas numpy
```

### 5.3 Frontend Dashboard Setup
```bash
cd frontend

# Install Node dependencies
npm install

# Start Vite development server
npm run dev
# Dashboard launches at http://localhost:5173
```

### 5.4 Running Benchmarks & Generating Paper Figures
```bash
# Run HumanEval benchmark simulation
python benchmarks/humaneval_eval.py --samples 50 --trip-threshold 0.65

# Run MBPP benchmark simulation
python benchmarks/mbpp_eval.py --samples 50

# Generate 300 DPI research figures for publication
python paper_results/generate_paper_charts.py
```

---

## 6. Citation & Research Attribution

If you use Epistemic Circuit Breaker in your academic work, please cite:

```bibtex
@article{epistemic_circuit_breaker_2026,
  title={Epistemic Circuit Breakers: Real-Time Detection and Rollback of Cascading Hallucinations in Multi-Agent LLM Systems},
  author={Epistemic Reliability & Multi-Agent Safety Research Group},
  journal={arXiv preprint arXiv:2603.xxxxx},
  year={2026}
}
```
