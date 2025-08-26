#!/usr/bin/env bash
set -euo pipefail

# helpful defaults if not provided
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

echo ">>> Starting FastAPI with SSE + vLLM embedded"
uvicorn app.main:app --host 0.0.0.0 --port 8000
