#!/usr/bin/env bash
set -euo pipefail

# Ensure conda Python is on PATH (vllm images use /opt/conda)
export PATH="/opt/conda/bin:${PATH}"

# Choose a python binary that exists
PY_BIN="$(command -v python3 || true)"
if [ -z "${PY_BIN}" ]; then
  PY_BIN="$(command -v python || true)"
fi
if [ -z "${PY_BIN}" ]; then
  echo "FATAL: Python not found in PATH" >&2
  exit 1
fi

# caches
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

# ports
export PORT="${PORT:-8000}"           # FastAPI
export VLLM_PORT="${VLLM_PORT:-8001}" # vLLM OpenAI server

# model args
export LLM_MODEL="${LLM_MODEL:-hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4}"
export LLM_QUANTIZATION="${LLM_QUANTIZATION:-awq}"
export VLLM_TP="${VLLM_TP:-1}"
export VLLM_MAX_LEN="${VLLM_MAX_LEN:-4096}"  # lower helps voice latency

echo ">>> Using Python at: ${PY_BIN}"
echo ">>> Starting vLLM OpenAI server on :$VLLM_PORT (model=$LLM_MODEL, quant=$LLM_QUANTIZATION)"
"${PY_BIN}" -m vllm.entrypoints.openai.api_server \
  --model "$LLM_MODEL" \
  ${LLM_QUANTIZATION:+--quantization "$LLM_QUANTIZATION"} \
  --tensor-parallel-size "$VLLM_TP" \
  --max-model-len "$VLLM_MAX_LEN" \
  --host 0.0.0.0 --port "$VLLM_PORT" &

echo ">>> Waiting for vLLM..."
for i in {1..120}; do
  if curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null; then break; fi
  sleep 1
done

echo ">>> Starting FastAPI on :$PORT"
exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
