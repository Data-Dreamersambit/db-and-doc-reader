#!/usr/bin/env bash
# ==============================================================================
# AgenticInsight - Unified Start Script
# Runs FastAPI Backend (Port 8000) and React/Vite Frontend (Port 5173) Concurrently
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ANSI Color Codes
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}======================================================${NC}"
echo -e "${CYAN}             Starting AgenticInsight Local            ${NC}"
echo -e "${CYAN}======================================================${NC}"

# 1. Check Python Virtual Environment
if [ -d ".venv" ]; then
    PYTHON_BIN=".venv/bin/python"
    UVICORN_BIN=".venv/bin/uvicorn"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
    UVICORN_BIN="uvicorn"
else
    echo -e "${RED}[ERROR] Python 3 not found. Please create a virtual environment (.venv).${NC}"
    exit 1
fi

# 2. Check Node.js / NPM
if ! command -v npm >/dev/null 2>&1; then
    echo -e "${RED}[ERROR] npm is not installed or not in PATH. Node.js 20+ required.${NC}"
    exit 1
fi

# 3. Check Ollama Status
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo -e "${YELLOW}[WARNING] Local Ollama server is not reachable at http://localhost:11434.${NC}"
    echo -e "${YELLOW}          Starting Ollama server in background...${NC}"
    if command -v ollama >/dev/null 2>&1; then
        ollama serve >/dev/null 2>&1 &
        sleep 2
    else
        echo -e "${RED}[ERROR] Ollama binary not found. Please install Ollama from https://ollama.ai${NC}"
    fi
fi

# Ensure storage directories exist
mkdir -p data sample_data evals

# Trap cleanup on exit or termination
cleanup() {
    echo -e "\n${YELLOW}Shutting down AgenticInsight services...${NC}"
    if [ -n "$BACKEND_PID" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        kill "$BACKEND_PID" 2>/dev/null || true
    fi
    if [ -n "$FRONTEND_PID" ] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
        kill "$FRONTEND_PID" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
    echo -e "${GREEN}All services stopped cleanly.${NC}"
}
trap cleanup EXIT INT TERM

# 4. Start FastAPI Backend
echo -e "${GREEN}--> Launching FastAPI Backend on http://localhost:8000...${NC}"
export PYTHONPATH="backend:$PYTHONPATH"
"$UVICORN_BIN" backend.app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend health check
sleep 2

# 5. Start React + Vite Frontend
echo -e "${GREEN}--> Launching Vite + React Frontend on http://localhost:5173...${NC}"
(cd frontend && npm run dev) &
FRONTEND_PID=$!

echo -e "\n${CYAN}======================================================${NC}"
echo -e "${GREEN} AgenticInsight is running successfully!${NC}"
echo -e "   - Frontend UI:   ${CYAN}http://localhost:5173${NC}"
echo -e "   - Backend API:   ${CYAN}http://localhost:8000${NC}"
echo -e "   - Swagger Docs:  ${CYAN}http://localhost:8000/docs${NC}"
echo -e "${CYAN}======================================================${NC}"
echo -e "Press ${YELLOW}[Ctrl+C]${NC} to stop both services.\n"

# Wait for both processes
wait
