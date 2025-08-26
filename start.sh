#!/usr/bin/env bash
set -euo pipefail

# caches (persist via RunPod volume)
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

# ports
export PORT="${PORT:-8000}"
export VLLM_PORT="${VLLM_PORT:-8001}"

# model args
export LLM_MODEL="${LLM_MODEL:-hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4}"
export LLM_QUANTIZATION="${LLM_QUANTIZATION:-awq}"
export VLLM_TP="${VLLM_TP:-1}"
export VLLM_MAX_LEN="${VLLM_MAX_LEN:-8192}"

echo ">>> Starting vLLM OpenAI server on :$VLLM_PORT (model=$LLM_MODEL, quant=$LLM_QUANTIZATION)"
python -m vllm.entrypoints.openai.api_server \
  --model "$LLM_MODEL" \
  ${LLM_QUANTIZATION:+--quantization "$LLM_QUANTIZATION"} \
  --tensor-parallel-size "$VLLM_TP" \
  --max-model-len "$VLLM_MAX_LEN" \
  --host 0.0.0.0 --port "$VLLM_PORT" &

# simple wait loop (avoid racing FastAPI before vLLM is up)
echo ">>> Waiting for vLLM to come up..."
for i in {1..120}; do
  if curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null; then
    break
  fi
  sleep 1
done

echo ">>> Starting FastAPI on :$PORT"
exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
