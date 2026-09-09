#!/usr/bin/env bash
# Run the CBPA case study — from quick deterministic check to full integrated demo.
#
# Usage:
#   bash scripts/run_full_demo.sh                                       # full demo (Claude, 3 shifts, macro, SimPy)
#   bash scripts/run_full_demo.sh --with-dashboard                      # + live Streamlit dashboard
#   bash scripts/run_full_demo.sh --no-llm                              # deterministic (no LLM calls)
#   bash scripts/run_full_demo.sh --no-llm --with-dashboard             # deterministic + dashboard
#   bash scripts/run_full_demo.sh --provider openai                     # use GPT instead of Claude
#   bash scripts/run_full_demo.sh --provider openai --model gpt-4o-mini # specific OpenAI model
#   bash scripts/run_full_demo.sh --integrated                          # Isaac Sim + BaSyx + OPC-UA (headless)
#   bash scripts/run_full_demo.sh --integrated --isaac-ui               # + Isaac Sim 3D viewport
#   bash scripts/run_full_demo.sh --integrated --isaac-ui --with-dashboard  # full visual demo
#   bash scripts/run_full_demo.sh --no-simulate                         # analytical only (no SimPy/Isaac)
#   bash scripts/run_full_demo.sh --shifts 5                            # custom number of shifts
#   bash scripts/run_full_demo.sh --output-dir /path/to/results         # custom output directory
#   bash scripts/run_full_demo.sh --factory                             # multi-cell factory experiment (5-round CBPA lifecycle)
#   bash scripts/run_full_demo.sh --factory --no-llm                    # factory deterministic mode
#   bash scripts/run_full_demo.sh --factory --supply-delay              # factory + V_C supply delay disturbance
#   bash scripts/run_full_demo.sh --factory --no-operator-absence       # factory without operator absence
#   bash scripts/run_full_demo.sh --factory --integrated                # factory + Isaac Sim + BaSyx + OPC-UA
#   bash scripts/run_full_demo.sh --factory --integrated --isaac-ui     # factory + Isaac Sim 3D viewport
#   bash scripts/run_full_demo.sh --factory --integrated --with-dashboard  # factory full visual demo
#
# Flags:
#   --no-llm              Deterministic mode (no LLM API calls, fastest)
#   --provider X          LLM provider: claude (CLI auth) or openai (API key)
#   --model X             LLM model override (e.g. gpt-4o-mini, sonnet)
#   --with-dashboard      Launch Streamlit dashboard with live phase updates
#   --integrated          Use Isaac Sim + BaSyx AAS + OPC-UA for execution
#   --isaac-ui            Open Isaac Sim 3D viewport (requires --integrated)
#   --no-simulate         Skip SimPy simulation (use analytical evaluation only)
#   --no-demo             Run without --full-demo flag (single shift, no macro)
#   --shifts N            Number of shifts (default: 3)
#   --output-dir DIR      Output directory for results and figures
#   --factory             Run multi-cell factory experiment (2 cells, 4 robots, 3 operators)
#   --supply-delay        [factory] Enable V_C supply delay disturbance
#   --no-operator-absence [factory] Disable operator absence disturbance
#
# Prerequisites:
#   pip install -e ".[dev,export]"            # base install
#   pip install -e ".[openai]"                # if using --provider openai
#   pip install -e ".[ui]"                    # if using --with-dashboard
#   claude login                              # if using Claude (default)
#   export OPENAI_API_KEY=sk-...              # if using OpenAI
#   Docker + NVIDIA GPU + conda env "isaac"   # if using --integrated

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── Defaults ───────────────────────────────────────────────────────
PROVIDER="claude"
MODEL=""
NO_LLM=false
SHIFTS=3
FULL_DEMO=true
SIMULATE=true
INTEGRATED=false
WITH_DASHBOARD=false
OUTPUT_DIR=""
FACTORY=false
SUPPLY_DELAY=false
NO_OPERATOR_ABSENCE=false
EXTRA_ARGS=()

# ── Parse arguments ────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --provider)            PROVIDER="$2"; shift 2 ;;
        --model)               MODEL="$2"; shift 2 ;;
        --no-llm)              NO_LLM=true; FULL_DEMO=false; shift ;;
        --shifts)              SHIFTS="$2"; shift 2 ;;
        --no-demo)             FULL_DEMO=false; shift ;;
        --no-simulate)         SIMULATE=false; shift ;;
        --integrated)          INTEGRATED=true; SIMULATE=false; shift ;;
        --isaac-ui)            ISAAC_UI=true; shift ;;
        --with-dashboard)      WITH_DASHBOARD=true; shift ;;
        --output-dir)          OUTPUT_DIR="$2"; shift 2 ;;
        --factory)             FACTORY=true; shift ;;
        --supply-delay)        SUPPLY_DELAY=true; shift ;;
        --no-operator-absence) NO_OPERATOR_ABSENCE=true; shift ;;
        *)                     EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ── Port helpers ──────────────────────────────────────────────────
kill_port() {
    local port=$1
    local pids
    pids=$(lsof -ti:"$port" 2>/dev/null) || true
    if [ -n "$pids" ]; then
        echo "  Killing stale process on port $port (PIDs: $pids)"
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 0.5
    fi
}

# ── Cleanup ────────────────────────────────────────────────────────
STREAMLIT_PID=""
INFRA_PID=""
cleanup() {
    if [ -n "$STREAMLIT_PID" ] && kill -0 "$STREAMLIT_PID" 2>/dev/null; then
        echo ""
        echo "  Stopping Streamlit dashboard (PID $STREAMLIT_PID)..."
        kill "$STREAMLIT_PID" 2>/dev/null || true
        wait "$STREAMLIT_PID" 2>/dev/null || true
    fi
    if [ -n "$INFRA_PID" ] && kill -0 "$INFRA_PID" 2>/dev/null; then
        echo ""
        echo "  Stopping integrated infrastructure (PID $INFRA_PID)..."
        kill "$INFRA_PID" 2>/dev/null || true
        wait "$INFRA_PID" 2>/dev/null || true
    fi
    # Stop Docker containers if we started them
    if [ "$INTEGRATED" = true ]; then
        echo "  Stopping Docker containers..."
        if docker info &>/dev/null; then
            docker compose -f "$SCRIPT_DIR/../docker/docker-compose.yml" down 2>/dev/null || true
        else
            sudo docker compose -f "$SCRIPT_DIR/../docker/docker-compose.yml" down 2>/dev/null || true
        fi

        # Final sweep: release any ports that survived cleanup
        echo "  Final port sweep..."
        for port in 8211 4840 8519 9081 9082 9083 9084 3000; do
            local_pids=$(lsof -ti:"$port" 2>/dev/null) || true
            if [ -n "$local_pids" ]; then
                echo "    Releasing port $port (PIDs: $local_pids)"
                echo "$local_pids" | xargs kill -9 2>/dev/null || true
            fi
        done
    fi
}
trap cleanup EXIT INT TERM

# ── Preflight checks ──────────────────────────────────────────────
echo "=================================================="
echo "  CBPA Case Study Runner"
echo "=================================================="
echo ""

# Check Python package is installed
if ! python -c "import cbpa" 2>/dev/null; then
    echo "ERROR: cbpa package not installed."
    echo "  Run:  pip install -e \".[dev,export]\""
    exit 1
fi

# Provider-specific checks
if [ "$NO_LLM" = false ]; then
    if [ "$PROVIDER" = "openai" ]; then
        # Check openai package
        if ! python -c "import openai" 2>/dev/null; then
            echo "ERROR: openai package not installed."
            echo "  Run:  pip install -e \".[openai]\""
            exit 1
        fi
        # Check API key
        if [ -z "${OPENAI_API_KEY:-}" ]; then
            echo "ERROR: OPENAI_API_KEY not set."
            echo "  Run:  export OPENAI_API_KEY=sk-your-key-here"
            exit 1
        fi
        echo "  Provider:  OpenAI (API key)"
        echo "  Model:     ${MODEL:-gpt-4o}"
    else
        echo "  Provider:  Claude (CLI auth)"
        echo "  Model:     ${MODEL:-claude-sonnet-4-6}"
    fi
else
    echo "  Mode:      Deterministic (no LLM)"
fi

echo "  Factory:    $FACTORY"
if [ "$FACTORY" = false ]; then
echo "  Shifts:     $SHIFTS"
echo "  Full demo:  $FULL_DEMO"
echo "  Simulate:   $SIMULATE"
fi
echo "  Integrated: $INTEGRATED"
echo "  Dashboard:  $WITH_DASHBOARD"
echo "  Output:     ${OUTPUT_DIR:-auto (from flags)}"
echo ""

# ── Launch integrated infrastructure (if requested) ───────────────
if [ "$INTEGRATED" = true ]; then
    echo "  Clearing stale processes on integrated ports..."
    kill_port 8211   # Isaac Sim
    kill_port 4840   # OPC-UA
    kill_port 9081   # BaSyx AAS server
    kill_port 9082   # BaSyx AAS registry
    kill_port 9083   # BaSyx AAS discovery
    kill_port 9084   # BaSyx submodel registry
    kill_port 3000   # BaSyx web UI
    # Cache sudo credentials now (before backgrounding) so Docker works
    if ! docker info &>/dev/null; then
        echo "  Docker requires sudo. Enter password now:"
        sudo -v || { echo "ERROR: sudo required for Docker."; exit 1; }
    fi

    echo "  Starting integrated infrastructure (Docker + Isaac Sim)..."
    INFRA_ARGS="--no-streamlit"
    if [ "${ISAAC_UI:-false}" = true ]; then
        INFRA_ARGS="$INFRA_ARGS --isaac-ui"
    fi
    if [ "$FACTORY" = true ]; then
        INFRA_ARGS="$INFRA_ARGS --multi-cell"
    fi
    bash "$SCRIPT_DIR/launch_integrated.sh" $INFRA_ARGS &
    INFRA_PID=$!

    # Wait for Isaac Sim REST API
    echo "  Waiting for Isaac Sim to be ready..."
    for i in $(seq 1 30); do
        if curl -s http://localhost:8211/status > /dev/null 2>&1; then
            echo "  Isaac Sim ready."
            break
        fi
        if [ "$i" -eq 30 ]; then
            echo "  WARNING: Isaac Sim not reachable (will use stub mode)."
        fi
        sleep 2
    done

    # Wait for BaSyx AAS registry
    echo "  Waiting for BaSyx AAS to be ready..."
    for i in $(seq 1 12); do
        if curl -s http://localhost:9082/shell-descriptors > /dev/null 2>&1; then
            echo "  BaSyx AAS ready."
            break
        fi
        if [ "$i" -eq 12 ]; then
            echo "  WARNING: BaSyx AAS not reachable (will use stub mode)."
        fi
        sleep 5
    done
    echo ""
fi

# ── Launch Streamlit dashboard (if requested) ─────────────────────
if [ "$WITH_DASHBOARD" = true ]; then
    # Check streamlit is installed
    if ! python -c "import streamlit" 2>/dev/null; then
        echo "ERROR: streamlit not installed."
        echo "  Run:  pip install -e \".[ui]\""
        exit 1
    fi

    DASHBOARD_PORT=8519

    # Free stale dashboard and OPC-UA ports
    kill_port "$DASHBOARD_PORT"
    kill_port 4840   # OPC-UA server started by Streamlit

    echo "  Starting Streamlit dashboard on port $DASHBOARD_PORT ..."
    STREAMLIT_ARGS=(streamlit run streamlit_app.py
        --server.port "$DASHBOARD_PORT"
        --server.headless true)

    streamlit run streamlit_app.py \
        --server.port "$DASHBOARD_PORT" \
        --server.headless true \
        > /tmp/cbpa_streamlit.log 2>&1 &
    STREAMLIT_PID=$!

    # If factory mode, print the URL with ?factory=1 query param
    if [ "$FACTORY" = true ]; then
        DASHBOARD_URL="http://localhost:$DASHBOARD_PORT/?factory=1"
    else
        DASHBOARD_URL="http://localhost:$DASHBOARD_PORT"
    fi

    # Wait for Streamlit to be ready (try the root page)
    for i in $(seq 1 15); do
        if curl -s -o /dev/null -w '%{http_code}' "http://localhost:$DASHBOARD_PORT/" 2>/dev/null | grep -qE '200|302'; then
            echo "  Dashboard ready: $DASHBOARD_URL"
            break
        fi
        if ! kill -0 "$STREAMLIT_PID" 2>/dev/null; then
            echo "  ERROR: Streamlit failed to start. Check /tmp/cbpa_streamlit.log"
            cat /tmp/cbpa_streamlit.log | tail -20
            exit 1
        fi
        sleep 1
    done
    echo ""
fi

# ── Build command ──────────────────────────────────────────────────
if [ "$FACTORY" = true ]; then
    # ── Factory experiment (multi-cell, 7-phase) ──────────────────
    CMD=(python scripts/run_factory_experiment.py)

    if [ -n "$OUTPUT_DIR" ]; then
        CMD+=(--output-dir "$OUTPUT_DIR")
    fi

    if [ "$NO_LLM" = true ]; then
        CMD+=(--no-llm)
    fi

    if [ "$SUPPLY_DELAY" = true ]; then
        CMD+=(--supply-delay)
    fi

    if [ "$NO_OPERATOR_ABSENCE" = true ]; then
        CMD+=(--no-operator-absence)
    fi

    if [ "$INTEGRATED" = true ]; then
        CMD+=(--integrated)
    fi

    if [ "$WITH_DASHBOARD" = true ]; then
        CMD+=(--with-dashboard)
    fi
else
    # ── Single-cell experiment (original 5-phase) ─────────────────
    CMD=(python scripts/run_experiment.py)

    if [ -n "$OUTPUT_DIR" ]; then
        CMD+=(--output-dir "$OUTPUT_DIR")
    fi

    if [ "$NO_LLM" = true ]; then
        CMD+=(--no-llm)
    fi

    if [ "$FULL_DEMO" = true ]; then
        CMD+=(--full-demo)
    else
        CMD+=(--shifts "$SHIFTS")
    fi

    if [ "$NO_LLM" = false ]; then
        CMD+=(--provider "$PROVIDER")
        if [ -n "$MODEL" ]; then
            CMD+=(--model "$MODEL")
        fi
    fi

    if [ "$SIMULATE" = true ]; then
        CMD+=(--simulate)
    fi

    if [ "$INTEGRATED" = true ]; then
        CMD+=(--integrated)
    fi

    if [ "$WITH_DASHBOARD" = true ]; then
        CMD+=(--with-dashboard)
    fi
fi

CMD+=("${EXTRA_ARGS[@]}")

# ── Run ────────────────────────────────────────────────────────────
echo "Running:  ${CMD[*]}"
echo "--------------------------------------------------"
echo ""

"${CMD[@]}"

echo ""
echo "=================================================="
echo "  Done. Results saved (see output above for directory)."
if [ "$WITH_DASHBOARD" = true ]; then
    echo "  Dashboard still running at ${DASHBOARD_URL:-http://localhost:$DASHBOARD_PORT}"
    echo "  Press Ctrl+C to stop."
    echo "=================================================="
    # Keep alive so the user can browse the dashboard
    wait "$STREAMLIT_PID" 2>/dev/null || true
else
    echo "=================================================="
fi
