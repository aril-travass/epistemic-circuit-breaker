#!/usr/bin/env python3
"""
Epistemic Circuit Breaker - Unified Development Launcher
Launches the FastAPI Backend (Port 8000) and React Vite Frontend (Port 5173) with
health monitoring, automatic readiness polling, and clean cross-platform shutdown.

Usage:
    python start_app.py              # Launch both Backend & Frontend
    python start_app.py --check-only # Pre-flight environment check
    python start_app.py --backend-only
    python start_app.py --frontend-only
"""

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")


def find_python_executable():
    """Locates the project virtualenv Python interpreter."""
    candidates = [
        os.path.join(BACKEND_DIR, "venv", "Scripts", "python.exe"),  # Windows
        os.path.join(BACKEND_DIR, "venv", "bin", "python"),         # POSIX
        os.path.join(ROOT_DIR, "venv", "Scripts", "python.exe"),
        os.path.join(ROOT_DIR, "venv", "bin", "python"),
        sys.executable,
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return sys.executable


def check_environment():
    """Performs pre-flight checks on backend and frontend setups."""
    print("=" * 70)
    print("🔍 Epistemic Circuit Breaker Pre-Flight Check")
    print("=" * 70)

    py_exe = find_python_executable()
    print(f"[1/4] Python Interpreter: {py_exe}")

    # Check key backend packages
    test_cmd = [
        py_exe,
        "-c",
        "import fastapi, uvicorn, pydantic, langgraph; print('Core backend dependencies OK')",
    ]
    try:
        res = subprocess.run(test_cmd, capture_output=True, text=True, cwd=BACKEND_DIR, timeout=30)
        if res.returncode == 0:
            print("[2/4] Backend Python Packages: ✓ Verified")
        else:
            print(f"[2/4] Backend Python Packages Warning: {res.stderr.strip()}")
    except Exception as e:
        print(f"[2/4] Backend Python Packages Error: {e}")

    # Check node and frontend
    node_modules = os.path.join(FRONTEND_DIR, "node_modules")
    if os.path.isdir(node_modules):
        print(f"[3/4] Frontend Dependencies: ✓ node_modules exists ({node_modules})")
    else:
        print("[3/4] Frontend Dependencies Warning: node_modules not found, run 'npm install' in frontend/")

    # Check main files
    main_py = os.path.join(BACKEND_DIR, "main.py")
    app_jsx = os.path.join(FRONTEND_DIR, "src", "App.jsx")
    if os.path.isfile(main_py) and os.path.isfile(app_jsx):
        print("[4/4] Project Files: ✓ main.py and App.jsx present")
    else:
        print("[4/4] Project Files Warning: Key source files missing")

    print("=" * 70)
    print("✓ Pre-flight environment check complete.")
    print("=" * 70)
    return True


def wait_for_url(url, timeout_sec=25, label="Service"):
    """Polls a URL until it responds with HTTP 200 or timeout occurs."""
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "start-app-check"})
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if resp.status in (200, 304):
                    return True
        except (urllib.error.URLError, TimeoutError, ConnectionRefusedError, OSError):
            time.sleep(0.5)
    return False


def main():
    parser = argparse.ArgumentParser(description="Launch Epistemic Circuit Breaker Full-Stack App")
    parser.add_argument("--check-only", action="store_true", help="Run pre-flight environment verification and exit")
    parser.add_argument("--backend-only", action="store_true", help="Launch only FastAPI backend")
    parser.add_argument("--frontend-only", action="store_true", help="Launch only Vite frontend")
    args = parser.parse_args()

    if args.check_only:
        check_environment()
        sys.exit(0)

    py_exe = find_python_executable()
    procs = []

    def cleanup(signum=None, frame=None):
        print("\n\n🛑 Shutting down Epistemic Circuit Breaker services...")
        for p, name in procs:
            if p.poll() is None:
                print(f"   Terminating {name} (PID: {p.pid})...")
                try:
                    p.terminate()
                    p.wait(timeout=3)
                except Exception:
                    try:
                        p.kill()
                    except Exception:
                        pass
        print("✓ All processes terminated cleanly. Goodbye!")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print("\n" + "=" * 76)
    print(" ⚡ STARTING EPISTEMIC CIRCUIT BREAKER RESEARCH DASHBOARD")
    print("=" * 76)

    # 1. Start Backend Server
    if not args.frontend_only:
        backend_cmd = [
            py_exe,
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--log-level",
            "info",
        ]
        print(f"🚀 Spawning FastAPI Backend on http://127.0.0.1:8000 ...")
        backend_proc = subprocess.Popen(
            backend_cmd,
            cwd=BACKEND_DIR,
            stdout=subprocess.DEVNULL if os.name == "nt" else None,
            stderr=None,
        )
        procs.append((backend_proc, "FastAPI Backend"))

        print("   Waiting for backend health check (/api/health)...")
        if wait_for_url("http://127.0.0.1:8000/api/health", timeout_sec=15, label="Backend"):
            print("   ✓ Backend is online and ready!")
        else:
            print("   ⚠️ Backend health check timed out, proceeding anyway...")

    # 2. Start Frontend Server
    if not args.backend_only:
        # On Windows, npm is npm.cmd
        npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
        frontend_cmd = [npm_cmd, "run", "dev", "--", "--host", "127.0.0.1", "--port", "5173"]
        print(f"🚀 Spawning Vite Frontend on http://127.0.0.1:5173 ...")
        frontend_proc = subprocess.Popen(
            frontend_cmd,
            cwd=FRONTEND_DIR,
            stdout=subprocess.DEVNULL if os.name == "nt" else None,
            stderr=None,
        )
        procs.append((frontend_proc, "Vite Frontend"))

        print("   Waiting for frontend server...")
        if wait_for_url("http://127.0.0.1:5173", timeout_sec=15, label="Frontend"):
            print("   ✓ Frontend dashboard is online and ready!")
        else:
            print("   ⚠️ Frontend check timed out, proceeding anyway...")

    print("\n" + "╔" + "═" * 74 + "╗")
    print("║     EPISTEMIC CIRCUIT BREAKER: RESEARCH CONTROL-ROOM DASHBOARD           ║")
    print("╚" + "═" * 74 + "╝")
    print("  🌐 Dashboard UI:             http://localhost:5173")
    print("  ⚡ Backend API:              http://localhost:8000")
    print("  📚 Interactive API Docs:     http://localhost:8000/docs")
    print("  📡 SSE Telemetry Stream:     http://localhost:8000/api/stream")
    print("  " + "─" * 74)
    print("  Press Ctrl+C in this terminal to shut down both servers.\n")

    try:
        while True:
            time.sleep(1)
            for p, name in procs:
                if p.poll() is not None:
                    print(f"⚠️ Warning: {name} process exited with code {p.returncode}")
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()
