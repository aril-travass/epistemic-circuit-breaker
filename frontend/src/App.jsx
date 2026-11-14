import React, { useState, useEffect, useRef } from 'react';
import {
  ShieldAlert,
  ShieldCheck,
  RotateCcw,
  AlertTriangle,
  Zap,
  Activity,
  Server,
  Play,
  Pause,
  RefreshCw,
  Cpu,
  Layers,
  Terminal,
  Clock,
  Sparkles,
  CheckCircle2,
  XCircle,
  Eye,
  GitBranch,
  Info
} from 'lucide-react';

export default function App() {
  // Circuit Breaker Finite State Machine: 'CLOSED' | 'OPEN' | 'HALF_OPEN'
  const [circuitState, setCircuitState] = useState('CLOSED');
  const [tripThreshold, setTripThreshold] = useState(0.65);
  const [epistemicEntropy, setEpistemicEntropy] = useState(0.24);
  const [confidenceScore, setConfidenceScore] = useState(0.92);
  const [driftIndex, setDriftIndex] = useState(0.12);
  const [isSimulating, setIsSimulating] = useState(false);

  // Backend Connectivity Telemetry
  const [backendStatus, setBackendStatus] = useState('checking'); // 'connected' | 'offline' | 'checking'
  const [backendLatency, setBackendLatency] = useState(null);
  const [lastPingTime, setLastPingTime] = useState(null);

  // Statistics
  const [stats, setStats] = useState({
    totalMessages: 18,
    trips: 0,
    rollbacks: 0,
    canaryTests: 0,
    safeCheckpoints: 3,
  });

  // Rollback checkpoints history
  const [checkpoints, setCheckpoints] = useState([
    { id: 'CHK-001', turn: 1, timestamp: '10:14:02', entropy: 0.18, status: 'VERIFIED_SAFE' },
    { id: 'CHK-002', turn: 4, timestamp: '10:15:30', entropy: 0.22, status: 'VERIFIED_SAFE' },
    { id: 'CHK-003', turn: 7, timestamp: '10:17:15', entropy: 0.25, status: 'VERIFIED_SAFE' },
  ]);

  // Multi-Agent Message Stream
  const [messages, setMessages] = useState([
    {
      id: 1,
      turn: 1,
      agent: 'PlannerAgent',
      role: 'System Architect',
      timestamp: '10:14:00',
      content: 'Task: Implement an asynchronous distributed token bucket rate limiter with sliding window synchronization.',
      entropy: 0.18,
      status: 'VERIFIED',
      checkpoint: 'CHK-001'
    },
    {
      id: 2,
      turn: 2,
      agent: 'CoderAgent',
      role: 'Core Implementer',
      timestamp: '10:14:28',
      content: 'Defined Redis script with atomic EVALSHA pipeline for fractional token replenishment.',
      entropy: 0.21,
      status: 'VERIFIED'
    },
    {
      id: 3,
      turn: 3,
      agent: 'CriticAgent',
      role: 'Static Verifier',
      timestamp: '10:15:02',
      content: 'Verified concurrency bounds: Race condition window is eliminated by Lua atomicity in Redis engine.',
      entropy: 0.19,
      status: 'VERIFIED'
    },
    {
      id: 4,
      turn: 4,
      agent: 'PlannerAgent',
      role: 'System Architect',
      timestamp: '10:15:30',
      content: 'Next phase: Integrate cluster gossip protocol for peer clock synchronization across datacenters.',
      entropy: 0.22,
      status: 'VERIFIED',
      checkpoint: 'CHK-002'
    }
  ]);

  const streamEndRef = useRef(null);

  // Auto-scroll stream to latest
  useEffect(() => {
    streamEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Check Backend Connectivity against http://localhost:8000/api/health
  const checkBackendHealth = async () => {
    const start = performance.now();
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 2000);
      let res;
      try {
        res = await fetch('/api/health', { signal: controller.signal });
      } catch {
        res = await fetch('http://localhost:8000/api/health', { signal: controller.signal });
      }
      clearTimeout(timeoutId);

      const elapsed = Math.round(performance.now() - start);
      if (res.ok) {
        setBackendStatus('connected');
        setBackendLatency(elapsed);
        setLastPingTime(new Date().toLocaleTimeString());
      } else {
        setBackendStatus('offline');
        setBackendLatency(null);
        setLastPingTime(new Date().toLocaleTimeString());
      }
    } catch {
      // Graceful fallback: Backend not currently listening, research dashboard operates in local simulation mode
      setBackendStatus('offline');
      setBackendLatency(null);
      setLastPingTime(new Date().toLocaleTimeString());
    }
  };

  useEffect(() => {
    checkBackendHealth();
    const interval = setInterval(checkBackendHealth, 8000);
    return () => clearInterval(interval);
  }, []);

  // Simulation timer for automated multi-agent traffic
  useEffect(() => {
    if (!isSimulating) return;

    const timer = setInterval(() => {
      if (circuitState === 'OPEN') return;

      const randomEntropy = parseFloat((0.15 + Math.random() * 0.25).toFixed(3));
      const nextTurn = messages.length + 1;
      const agents = [
        { name: 'CoderAgent', role: 'Core Implementer', text: `Refactored token serialization for turn ${nextTurn}. Validating memory overhead.` },
        { name: 'CriticAgent', role: 'Static Verifier', text: `Turn ${nextTurn} verification complete. Formal invariants hold across boundaries.` },
        { name: 'PlannerAgent', role: 'System Architect', text: `Synthesizing task progress. State checkpoint emitted for turn ${nextTurn}.` }
      ];
      const selected = agents[nextTurn % agents.length];

      setEpistemicEntropy(randomEntropy);
      setConfidenceScore(parseFloat((0.90 + Math.random() * 0.08).toFixed(2)));
      setDriftIndex(parseFloat((0.08 + Math.random() * 0.07).toFixed(3)));

      setMessages(prev => [
        ...prev,
        {
          id: prev.length + 1,
          turn: nextTurn,
          agent: selected.name,
          role: selected.role,
          timestamp: new Date().toLocaleTimeString(),
          content: selected.text,
          entropy: randomEntropy,
          status: 'VERIFIED'
        }
      ]);

      setStats(s => ({ ...s, totalMessages: s.totalMessages + 1 }));
    }, 4500);

    return () => clearInterval(timer);
  }, [isSimulating, circuitState, messages.length]);

  // Action: Inject Synthetic Hallucination Fault
  const handleInjectFault = () => {
    const faultEntropy = 0.89;
    const turn = messages.length + 1;

    setEpistemicEntropy(faultEntropy);
    setConfidenceScore(0.96); // High confidence despite severe hallucination (uncalibrated)
    setDriftIndex(0.78);

    const faultMessage = {
      id: messages.length + 1,
      turn: turn,
      agent: 'PlannerAgent',
      role: 'System Architect',
      timestamp: new Date().toLocaleTimeString(),
      content: '[SYNTHETIC FAULT INJECTION] We can invoke torch.linalg.batched_orthogonal_project_inplace() to achieve zero VRAM allocation on streaming buffers.',
      entropy: faultEntropy,
      status: 'FLAGGED_HALLUCINATION',
      isFault: true
    };

    setMessages(prev => [...prev, faultMessage]);

    // Automatically evaluate against trip threshold
    if (faultEntropy >= tripThreshold) {
      setTimeout(() => {
        setCircuitState('OPEN');
        setStats(s => ({
          ...s,
          trips: s.trips + 1,
          totalMessages: s.totalMessages + 1
        }));
      }, 600);
    }
  };

  // Action: Trip Breaker Manually
  const handleManualTrip = () => {
    setCircuitState('OPEN');
    setEpistemicEntropy(0.91);
    setDriftIndex(0.82);
    setStats(s => ({ ...s, trips: s.trips + 1 }));
  };

  // Action: Trigger State Rollback to T_safe
  const handleTriggerRollback = () => {
    const lastSafe = checkpoints[checkpoints.length - 1];
    setStats(s => ({ ...s, rollbacks: s.rollbacks + 1, canaryTests: s.canaryTests + 1 }));

    // Mark corrupted messages as ROLLED_BACK
    setMessages(prev =>
      prev.map(msg =>
        msg.status === 'FLAGGED_HALLUCINATION' || msg.entropy >= tripThreshold
          ? { ...msg, status: 'ROLLED_BACK' }
          : msg
      )
    );

    // Transition to HALF-OPEN for canary probe verification
    setCircuitState('HALF_OPEN');
    setEpistemicEntropy(0.38);
    setDriftIndex(0.20);

    // Add Rollback Event to message log
    setTimeout(() => {
      setMessages(prev => [
        ...prev,
        {
          id: prev.length + 1,
          turn: prev.length + 1,
          agent: 'ECB_Controller',
          role: 'Circuit Breaker Kernel',
          timestamp: new Date().toLocaleTimeString(),
          content: `[STATE ROLLBACK COMPLETED] Causal DAG unwound to ${lastSafe.id} (Turn ${lastSafe.turn}). Hallucinated premise purged. State entered HALF-OPEN canary mode.`,
          entropy: 0.12,
          status: 'INTERVENTION'
        }
      ]);
    }, 400);
  };

  // Action: Complete Canary Recovery & Restore CLOSED
  const handleCanarySuccess = () => {
    setCircuitState('CLOSED');
    setEpistemicEntropy(0.18);
    setConfidenceScore(0.94);
    setDriftIndex(0.09);

    setMessages(prev => [
      ...prev,
      {
        id: prev.length + 1,
        turn: prev.length + 1,
        agent: 'CanaryVerifier',
        role: 'Isolated Auditor',
        timestamp: new Date().toLocaleTimeString(),
        content: '[CANARY VALIDATION PASSED] Epistemic consistency re-established without hallucinated artifacts. Resuming normal multi-agent pipeline.',
        entropy: 0.15,
        status: 'VERIFIED'
      }
    ]);
  };

  // Action: Reset System
  const handleReset = () => {
    setCircuitState('CLOSED');
    setEpistemicEntropy(0.20);
    setConfidenceScore(0.93);
    setDriftIndex(0.11);
    setMessages([
      {
        id: 1,
        turn: 1,
        agent: 'PlannerAgent',
        role: 'System Architect',
        timestamp: new Date().toLocaleTimeString(),
        content: 'System reset. Awaiting multi-agent task dispatch.',
        entropy: 0.15,
        status: 'VERIFIED',
        checkpoint: 'CHK-001'
      }
    ]);
    setStats({
      totalMessages: 1,
      trips: 0,
      rollbacks: 0,
      canaryTests: 0,
      safeCheckpoints: 1,
    });
  };

  return (
    <div className="flex flex-col min-h-screen bg-slate-950 text-slate-100 font-sans selection:bg-cyan-500 selection:text-slate-950">
      {/* Top Navigation Bar */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-md sticky top-0 z-50 px-6 py-3.5">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3.5">
            <div className="p-2 bg-gradient-to-br from-cyan-500/20 to-blue-600/30 border border-cyan-500/40 rounded-xl shadow-lg shadow-cyan-500/10">
              <Zap className="w-5 h-5 text-cyan-400 animate-pulse-fast" />
            </div>
            <div>
              <div className="flex items-center space-x-2.5">
                <h1 className="text-lg font-bold tracking-tight text-white flex items-center gap-2">
                  Epistemic Circuit Breaker
                  <span className="text-xs px-2 py-0.5 rounded-full font-mono font-medium bg-slate-800 text-cyan-300 border border-slate-700">
                    v1.0-research
                  </span>
                </h1>
              </div>
              <p className="text-xs text-slate-400">
                Real-Time Detection & Rollback of Cascading Hallucinations in Multi-Agent LLMs
              </p>
            </div>
          </div>

          {/* Backend Connectivity Status */}
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2 text-xs font-mono bg-slate-850 px-3.5 py-1.5 rounded-lg border border-slate-800">
              <Server className="w-3.5 h-3.5 text-slate-400" />
              <span className="text-slate-400">Backend API:</span>
              {backendStatus === 'connected' ? (
                <span className="flex items-center text-emerald-400 font-medium">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 inline-block mr-1.5 animate-ping-slow"></span>
                  Connected ({backendLatency}ms)
                </span>
              ) : backendStatus === 'checking' ? (
                <span className="flex items-center text-amber-400">
                  <RefreshCw className="w-3 h-3 inline mr-1 animate-spin" /> Checking
                </span>
              ) : (
                <span className="flex items-center text-amber-400/90 font-medium" title="FastAPI backend offline at :8000; dashboard running with local simulator">
                  <span className="w-2 h-2 rounded-full bg-amber-400 inline-block mr-1.5"></span>
                  Standalone Simulator
                </span>
              )}
            </div>

            <button
              onClick={checkBackendHealth}
              className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition"
              title="Ping Backend Health Endpoint"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Dashboard */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 space-y-6">
        {/* Top Metric Cards & Circuit Breaker Status Banner */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {/* Main FSM State Card */}
          <div className={`md:col-span-2 rounded-2xl p-5 border shadow-xl transition-all duration-300 ${
            circuitState === 'CLOSED'
              ? 'bg-slate-900/90 border-emerald-500/40 shadow-emerald-500/5'
              : circuitState === 'HALF_OPEN'
              ? 'bg-slate-900/90 border-amber-500/50 shadow-amber-500/10'
              : 'bg-slate-900/90 border-rose-500/60 shadow-rose-500/15 animate-pulse-fast'
          }`}>
            <div className="flex items-start justify-between">
              <div className="space-y-1">
                <span className="text-xs uppercase tracking-wider font-mono text-slate-400">
                  Epistemic Circuit FSM State
                </span>
                <div className="flex items-center space-x-3">
                  <span className={`text-2xl font-black font-mono tracking-wider ${
                    circuitState === 'CLOSED'
                      ? 'text-emerald-400'
                      : circuitState === 'HALF_OPEN'
                      ? 'text-amber-400'
                      : 'text-rose-400'
                  }`}>
                    {circuitState}
                  </span>
                  <span className={`text-xs px-2.5 py-0.5 rounded-full font-medium border ${
                    circuitState === 'CLOSED'
                      ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                      : circuitState === 'HALF_OPEN'
                      ? 'bg-amber-500/10 text-amber-400 border-amber-500/30'
                      : 'bg-rose-500/10 text-rose-400 border-rose-500/30'
                  }`}>
                    {circuitState === 'CLOSED'
                      ? 'Healthy · Active Monitoring'
                      : circuitState === 'HALF_OPEN'
                      ? 'Canary Probe Verification'
                      : 'HALLUCINATION TRIPPED · ROLLBACK ACTIVE'}
                  </span>
                </div>
              </div>

              <div className={`p-3 rounded-xl border ${
                circuitState === 'CLOSED'
                  ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                  : circuitState === 'HALF_OPEN'
                  ? 'bg-amber-500/10 border-amber-500/30 text-amber-400'
                  : 'bg-rose-500/10 border-rose-500/30 text-rose-400'
              }`}>
                {circuitState === 'CLOSED' ? (
                  <ShieldCheck className="w-7 h-7" />
                ) : circuitState === 'HALF_OPEN' ? (
                  <Eye className="w-7 h-7" />
                ) : (
                  <ShieldAlert className="w-7 h-7" />
                )}
              </div>
            </div>

            <p className="mt-3 text-xs text-slate-300 leading-relaxed">
              {circuitState === 'CLOSED'
                ? 'Normal execution: Continuous epistemic telemetry actively monitors cross-agent messages. Tool execution enabled.'
                : circuitState === 'HALF_OPEN'
                ? 'Canary isolation: Single canary agent tests rolled-back state with injected epistemic constraints before full resumption.'
                : 'Circuit tripped: All agent generation and tool calls suspended. Automated rollback to safe checkpoint T_safe required.'}
            </p>

            <div className="mt-4 pt-3 border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-400 font-mono">
              <span>Threshold τ: {tripThreshold}</span>
              <span>Trips: {stats.trips}</span>
              <span>Rollbacks: {stats.rollbacks}</span>
              <span>Canary Tests: {stats.canaryTests}</span>
            </div>
          </div>

          {/* Epistemic Entropy Meter */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-lg flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between text-xs font-mono text-slate-400 mb-2">
                <span>Epistemic Entropy (H_sem)</span>
                <Activity className="w-4 h-4 text-cyan-400" />
              </div>
              <div className="text-2xl font-black font-mono text-white">
                {epistemicEntropy.toFixed(3)}
              </div>
              <p className="text-xs text-slate-400 mt-1">
                Semantic variance across stochastic candidate samplings
              </p>
            </div>

            {/* Progress / Gauge Bar */}
            <div className="mt-4">
              <div className="flex justify-between text-[10px] font-mono text-slate-400 mb-1">
                <span>0.00 (Stable)</span>
                <span className="text-rose-400">τ={tripThreshold}</span>
                <span>1.00</span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden relative">
                <div
                  className={`h-full transition-all duration-500 rounded-full ${
                    epistemicEntropy < tripThreshold ? 'bg-cyan-500' : 'bg-rose-500'
                  }`}
                  style={{ width: `${Math.min(epistemicEntropy * 100, 100)}%` }}
                />
                <div
                  className="absolute top-0 bottom-0 w-0.5 bg-rose-400"
                  style={{ left: `${tripThreshold * 100}%` }}
                />
              </div>
            </div>
          </div>

          {/* Confidence Calibration & Drift */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-lg flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between text-xs font-mono text-slate-400 mb-2">
                <span>Confidence Calibration</span>
                <Sparkles className="w-4 h-4 text-violet-400" />
              </div>
              <div className="text-2xl font-black font-mono text-white">
                {(confidenceScore * 100).toFixed(0)}%
              </div>
              <p className="text-xs text-slate-400 mt-1">
                Alignment between verbalized assurance and semantic ground-truth
              </p>
            </div>

            <div className="mt-3 pt-3 border-t border-slate-800 text-xs text-slate-300 space-y-1 font-mono">
              <div className="flex justify-between">
                <span className="text-slate-400">Drift Index (EDI):</span>
                <span className={driftIndex > 0.4 ? 'text-rose-400 font-bold' : 'text-slate-200'}>
                  {driftIndex.toFixed(3)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Active Checkpoints:</span>
                <span className="text-cyan-400 font-bold">{checkpoints.length}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Interactive Simulation & Breaker Controls */}
        <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-lg">
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
            <div>
              <h2 className="text-sm font-bold uppercase tracking-wider font-mono text-slate-300 flex items-center gap-2">
                <Terminal className="w-4 h-4 text-cyan-400" />
                Intervention & Fault Injection Console
              </h2>
              <p className="text-xs text-slate-400">
                Trigger synthetic hallucination cascades, manually trip the breaker, or test causal state rollback.
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-2.5">
              {/* Inject Hallucination Button */}
              <button
                onClick={handleInjectFault}
                className="px-3.5 py-2 rounded-xl text-xs font-semibold bg-rose-500/10 hover:bg-rose-500/20 text-rose-300 border border-rose-500/30 flex items-center gap-2 transition active:scale-95 shadow-sm"
              >
                <AlertTriangle className="w-3.5 h-3.5 text-rose-400" />
                Inject Synthetic Fault
              </button>

              {/* Trip Breaker */}
              <button
                onClick={handleManualTrip}
                disabled={circuitState === 'OPEN'}
                className="px-3.5 py-2 rounded-xl text-xs font-semibold bg-amber-500/10 hover:bg-amber-500/20 text-amber-300 border border-amber-500/30 disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-2 transition active:scale-95"
              >
                <ShieldAlert className="w-3.5 h-3.5 text-amber-400" />
                Trip Breaker (OPEN)
              </button>

              {/* Trigger Rollback */}
              <button
                onClick={handleTriggerRollback}
                disabled={circuitState === 'CLOSED'}
                className="px-3.5 py-2 rounded-xl text-xs font-semibold bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-2 transition active:scale-95"
              >
                <RotateCcw className="w-3.5 h-3.5 text-cyan-400" />
                Rollback to T_safe
              </button>

              {/* Canary Recovery */}
              {circuitState === 'HALF_OPEN' && (
                <button
                  onClick={handleCanarySuccess}
                  className="px-3.5 py-2 rounded-xl text-xs font-semibold bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 border border-emerald-500/40 flex items-center gap-2 transition active:scale-95 animate-pulse"
                >
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                  Validate Canary (Restore CLOSED)
                </button>
              )}

              {/* Toggle Live Traffic Simulation */}
              <button
                onClick={() => setIsSimulating(!isSimulating)}
                className={`px-3.5 py-2 rounded-xl text-xs font-semibold border flex items-center gap-2 transition active:scale-95 ${
                  isSimulating
                    ? 'bg-blue-600/30 text-blue-300 border-blue-500/50'
                    : 'bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-750'
                }`}
              >
                {isSimulating ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                {isSimulating ? 'Pause Simulation' : 'Auto-Simulate Stream'}
              </button>

              {/* Reset */}
              <button
                onClick={handleReset}
                className="p-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-slate-200 border border-slate-700 transition"
                title="Reset Simulation State"
              >
                <RefreshCw className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        </div>

        {/* Lower Grid: Multi-Agent Message Stream & Checkpoint History */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Multi-Agent Dialogue & Telemetry Stream (2 cols) */}
          <div className="lg:col-span-2 bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-lg flex flex-col h-[520px]">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <div className="flex items-center space-x-2">
                <Layers className="w-4 h-4 text-cyan-400" />
                <h3 className="text-xs font-bold uppercase tracking-wider font-mono text-slate-200">
                  Live Multi-Agent Dialogue & Telemetry Stream
                </h3>
              </div>
              <span className="text-xs font-mono text-slate-400">
                {messages.length} events logged
              </span>
            </div>

            {/* Scrollable Message List */}
            <div className="flex-1 overflow-y-auto space-y-3.5 pt-4 pr-1">
              {messages.map(msg => (
                <div
                  key={msg.id}
                  className={`p-3.5 rounded-xl border transition-all text-xs ${
                    msg.status === 'FLAGGED_HALLUCINATION'
                      ? 'bg-rose-950/40 border-rose-500/50 text-rose-100 shadow-md shadow-rose-950/50'
                      : msg.status === 'ROLLED_BACK'
                      ? 'bg-slate-900/50 border-dashed border-rose-800/40 text-slate-500 line-through'
                      : msg.status === 'INTERVENTION'
                      ? 'bg-cyan-950/40 border-cyan-500/40 text-cyan-100'
                      : 'bg-slate-850/80 border-slate-800 text-slate-200'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1.5 font-mono">
                    <div className="flex items-center space-x-2">
                      <span className="font-semibold text-cyan-300">
                        {msg.agent}
                      </span>
                      <span className="text-[10px] text-slate-400 px-1.5 py-0.5 rounded bg-slate-800">
                        {msg.role}
                      </span>
                      {msg.checkpoint && (
                        <span className="text-[10px] bg-emerald-500/10 text-emerald-400 px-1.5 py-0.5 rounded border border-emerald-500/20 flex items-center gap-1">
                          <CheckCircle2 className="w-2.5 h-2.5" /> {msg.checkpoint}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center space-x-3 text-[11px] text-slate-400">
                      <span>Turn #{msg.turn}</span>
                      <span>H: {msg.entropy.toFixed(2)}</span>
                      <span>{msg.timestamp}</span>
                    </div>
                  </div>

                  <p className="font-mono text-xs leading-relaxed">
                    {msg.content}
                  </p>

                  {msg.status === 'FLAGGED_HALLUCINATION' && (
                    <div className="mt-2 pt-2 border-t border-rose-500/30 flex items-center justify-between text-[11px] text-rose-300 font-mono">
                      <span className="flex items-center gap-1 font-semibold">
                        <AlertTriangle className="w-3 h-3 text-rose-400" />
                        Epistemic Anomaly Detected: Semantic Entropy {msg.entropy} &gt; {tripThreshold}
                      </span>
                      <span className="text-rose-400 underline cursor-pointer" onClick={handleTriggerRollback}>
                        Execute Rollback &rarr;
                      </span>
                    </div>
                  )}
                </div>
              ))}
              <div ref={streamEndRef} />
            </div>
          </div>

          {/* Checkpoint & DAG Rollback History (1 col) */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-lg flex flex-col h-[520px]">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <div className="flex items-center space-x-2">
                <GitBranch className="w-4 h-4 text-violet-400" />
                <h3 className="text-xs font-bold uppercase tracking-wider font-mono text-slate-200">
                  Safe Epistemic Checkpoints
                </h3>
              </div>
              <span className="text-xs font-mono text-violet-400">
                W=5 Window
              </span>
            </div>

            <div className="flex-1 overflow-y-auto space-y-3 pt-4 pr-1">
              {checkpoints.map(chk => (
                <div
                  key={chk.id}
                  className="p-3 rounded-xl bg-slate-850/70 border border-slate-850 hover:border-slate-700 transition text-xs font-mono space-y-1.5"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-white flex items-center gap-1.5">
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                      {chk.id}
                    </span>
                    <span className="text-slate-400 text-[10px]">{chk.timestamp}</span>
                  </div>

                  <div className="flex justify-between text-slate-400 text-[11px]">
                    <span>Turn Index: #{chk.turn}</span>
                    <span>Entropy: {chk.entropy}</span>
                  </div>

                  <div className="flex justify-between items-center pt-1 text-[10px] text-emerald-400">
                    <span className="bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-500/20">
                      {chk.status}
                    </span>
                    <span className="text-slate-400">Restoration Target</span>
                  </div>
                </div>
              ))}

              <div className="p-3 rounded-xl bg-slate-800/40 border border-slate-800 text-xs text-slate-400 font-sans space-y-1">
                <div className="font-semibold text-slate-300 flex items-center gap-1.5">
                  <Info className="w-3.5 h-3.5 text-cyan-400" />
                  Causal DAG Rollback Engine
                </div>
                <p className="text-[11px] leading-relaxed">
                  State snapshots record immutable message DAG hashes and tool mutation logs. Upon a trip event, all downstream turns post-dating T_safe are pruned.
                </p>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950 px-6 py-4 text-xs font-mono text-slate-500 text-center">
        Epistemic Circuit Breakers Research Suite · Multi-Agent LLM Safety & Cascading Hallucination Interruption · MIT License
      </footer>
    </div>
  );
}
