#!/usr/bin/env bash
# Launch the full CBPA integrated system.
#
# Components started:
#   1. Docker: BaSyx AAS infrastructure (ports 9081-9084, 3000)
#   2. Isaac Sim: physics-based production cell (port 8211)
#   3. Streamlit: CBPA dashboard (port 8519)
#   4. OPC-UA: server starts automatically inside Streamlit (port 4840)
#
# Usage:
#   bash scripts/launch_integrated.sh
#   bash scripts/launch_integrated.sh --no-isaac    # skip Isaac Sim
#   bash scripts/launch_integrated.sh --no-docker   # skip Docker/BaSyx
#   bash scripts/launch_integrated.sh --env cbpa    # use a different conda env
#
# Prerequisites:
#   - conda env "isaac" (or specified env) with: isaacsim, streamlit, asyncua, httpx, cbpa
#   - Docker + Docker Compose (for BaSyx)
#   - NVIDIA GPU with driver >= 550 (for Isaac Sim)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

SKIP_ISAAC=false
SKIP_DOCKER=false
SKIP_STREAMLIT=false
ISAAC_UI=false
MULTI_CELL=false
CONDA_ENV="isaac"

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-isaac)     SKIP_ISAAC=true; shift ;;
        --no-docker)    SKIP_DOCKER=true; shift ;;
        --no-streamlit) SKIP_STREAMLIT=true; shift ;;
        --isaac-ui)     ISAAC_UI=true; shift ;;
        --multi-cell)   MULTI_CELL=true; shift ;;
        --env)          CONDA_ENV="$2"; shift 2 ;;
        *)              shift ;;
    esac
done

# Resolve conda run prefix
CONDA_RUN="conda run --no-capture-output -n $CONDA_ENV"

# Use sudo for docker if current user can't access docker socket
if docker info &>/dev/null; then
    DOCKER_CMD="docker"
else
    echo "Docker requires sudo. Requesting credentials..."
    # Cache sudo credentials upfront (before we go to background)
    sudo -v || { echo "ERROR: sudo required for Docker. Aborting."; exit 1; }
    DOCKER_CMD="sudo docker"
fi

# ── Pre-launch: kill anything already on our ports ───────────────────
kill_port() {
    local port=$1
    local pids
    pids=$(lsof -ti:"$port" 2>/dev/null) || true
    if [ -n "$pids" ]; then
        echo "  Killing existing process on port $port (PIDs: $pids)"
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 0.5
    fi
}

echo "Checking for stale processes on required ports..."
kill_port 8519   # Streamlit
kill_port 4840   # OPC-UA
kill_port 8211   # Isaac Sim
kill_port 9081   # BaSyx AAS server
kill_port 9082   # BaSyx AAS registry
kill_port 9083   # BaSyx AAS discovery
kill_port 9084   # BaSyx submodel registry
kill_port 3000   # BaSyx web UI

echo "╔══════════════════════════════════════════════════════════╗"
echo "║         CBPA Integrated System Launcher                 ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  BaSyx AAS:   http://localhost:3000  (Docker)           ║"
echo "║  Streamlit:   http://localhost:8519  (Dashboard)        ║"
echo "║  OPC-UA:      opc.tcp://localhost:4840/cbpa/ (auto)     ║"
echo "║  Isaac Sim:   http://localhost:8211  (physics)          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Track PIDs for cleanup
PIDS=()

cleanup() {
    echo ""
    echo "Shutting down..."

    # Kill all tracked child processes and their entire process trees
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "  Stopping process tree rooted at PID $pid..."
            # Kill the entire process group (children, grandchildren, etc.)
            pkill -TERM -P "$pid" 2>/dev/null || true
            kill -TERM "$pid" 2>/dev/null || true
            sleep 1
            # Force kill any survivors
            pkill -9 -P "$pid" 2>/dev/null || true
            kill -9 "$pid" 2>/dev/null || true
        fi
    done

    # Final sweep: only kill the LISTENING server on port 8211, NOT clients
    # connected to it. kill_port uses lsof which returns ALL PIDs (server +
    # clients), and kill -9 on a client PID kills the experiment process.
    # Instead, kill only the process that is LISTENING (state=LISTEN).
    local listen_pids
    listen_pids=$(lsof -ti:8211 -sTCP:LISTEN 2>/dev/null) || true
    if [ -n "$listen_pids" ]; then
        echo "  Killing Isaac Sim listener on port 8211 (PIDs: $listen_pids)"
        echo "$listen_pids" | xargs kill -9 2>/dev/null || true
    fi

    if [ "$SKIP_DOCKER" = false ]; then
        echo "  Stopping Docker containers..."
        $DOCKER_CMD compose -f "$PROJECT_DIR/docker/docker-compose.yml" down 2>/dev/null || true
    fi
    echo "Done. All ports released."
}
trap cleanup EXIT INT TERM

# ── 1. Docker: BaSyx infrastructure ──────────────────────────────────
if [ "$SKIP_DOCKER" = false ]; then
    echo "[1/3] Starting BaSyx Docker infrastructure..."
    if ! $DOCKER_CMD compose -f "$PROJECT_DIR/docker/docker-compose.yml" up -d 2>/dev/null; then
        echo "      WARNING: Docker compose failed (permission or service issue)."
        echo "      BaSyx will not be available. Continuing without it."
        SKIP_DOCKER=true
    fi
    echo "      Waiting for BaSyx services to be ready..."
    sleep 10
    # Check AAS registry
    for i in $(seq 1 12); do
        if curl -s http://localhost:9082/shell-descriptors > /dev/null 2>&1; then
            echo "      BaSyx AAS registry is ready."
            break
        fi
        echo "      Waiting... ($i/12)"
        sleep 5
    done
else
    echo "[1/3] Skipping Docker (--no-docker)"
fi

# ── 2. Isaac Sim ─────────────────────────────────────────────────────
if [ "$SKIP_ISAAC" = false ]; then
    echo "[2/3] Starting Isaac Sim (headless, port 8211)..."
    if ! conda info --envs 2>/dev/null | grep -q "$CONDA_ENV"; then
        echo "      WARNING: conda env '$CONDA_ENV' not found. Skipping Isaac Sim."
        SKIP_ISAAC=true
    else
        ISAAC_ARGS=""
        if [ "$ISAAC_UI" = true ]; then
            ISAAC_ARGS="--live-ui"
        fi
        if [ "$MULTI_CELL" = true ]; then
            ISAAC_ARGS="$ISAAC_ARGS --multi-cell"
        fi
        OMNI_KIT_ACCEPT_EULA=YES $CONDA_RUN python "$SCRIPT_DIR/isaac_scene_setup.py" $ISAAC_ARGS &
        PIDS+=($!)
        echo "      Isaac Sim PID: ${PIDS[-1]}"
        # Wait for Isaac REST API
        for i in $(seq 1 30); do
            if curl -s http://localhost:8211/status > /dev/null 2>&1; then
                echo "      Isaac Sim REST API is ready."
                break
            fi
            sleep 2
        done
    fi
else
    echo "[2/3] Skipping Isaac Sim (--no-isaac)"
fi

# ── 3. Streamlit dashboard ───────────────────────────────────────────
if [ "$SKIP_STREAMLIT" = false ]; then
    echo "[3/3] Starting Streamlit dashboard (port 8519)..."
    cd "$PROJECT_DIR"
    $CONDA_RUN streamlit run streamlit_app.py --server.port 8519 --server.headless true &
    PIDS+=($!)
    echo "      Streamlit PID: ${PIDS[-1]}"

    echo ""
    echo "All components launched. Open http://localhost:8519"
    echo "Press Ctrl+C to stop everything."
    echo ""

    # Wait for any child to exit
    wait
else
    echo "[3/3] Skipping Streamlit (--no-streamlit)"
    echo ""
    echo "Infrastructure launched (Docker + Isaac Sim). No dashboard."
    echo "Press Ctrl+C to stop everything."
    echo ""

    # Stay alive until the caller (run_full_demo.sh) kills us.
    # IMPORTANT: do NOT use `wait` here — if Isaac Sim crashes, `wait`
    # returns, this script exits, and the EXIT trap fires cleanup() which
    # kills processes connected to port 8211, including the experiment.
    # Instead, sleep forever; the caller sends SIGTERM when the experiment
    # is done, and the trap handles orderly shutdown.
    _KEEP_RUNNING=true
    trap '_KEEP_RUNNING=false' INT TERM
    trap cleanup EXIT
    if [ ${#PIDS[@]} -gt 0 ]; then
        # Sleep in short intervals; exit on signal or if parent dies
        while $_KEEP_RUNNING && kill -0 $PPID 2>/dev/null; do sleep 5 || true; done
    else
        # No background processes but Docker may be running.
        # Stay alive so the caller (run_full_demo.sh) can use
        # the infrastructure; cleanup runs when caller kills us.
        echo "Infrastructure services running. Waiting for signal..."
        # Sleep forever — cleanup trap fires when we're killed
        while true; do sleep 3600; done
    fi
fi
