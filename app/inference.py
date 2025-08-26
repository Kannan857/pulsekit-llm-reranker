from __future__ import annotations
import os, torch
from typing import List, Optional, Tuple
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams

load_dotenv()

llm_engine: Optional[AsyncLLMEngine] = None
reranker_model: Optional[CrossEncoder] = None

def _env(name: str, default: str | None = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v not in (None, "") else default

def load_models() -> None:
    """Initialize vLLM (quantized optional) + lightweight CPU reranker."""
    global llm_engine, reranker_model

    # ---- LLM via vLLM ----
    # AWQ INT4 (good for A10G 24GB etc.)
    model_id = _env("LLM_MODEL", "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4")
    quantization = _env("LLM_QUANTIZATION", "awq")   # set "" for FP model
    tp = int(_env("VLLM_TP", "1"))
    max_ctx = int(_env("VLLM_MAX_LEN", "8192"))
    gpu_util = float(_env("VLLM_GPU_UTIL", "0.90"))
    download_dir = _env("HF_HOME", None) or _env("TRANSFORMERS_CACHE", None)

    engine_args = AsyncEngineArgs(
        model=model_id,
        trust_remote_code=True,
        tensor_parallel_size=tp,
        max_model_len=max_ctx,
        dtype="auto",
        gpu_memory_utilization=gpu_util,
        quantization=quantization if quantization else None,
        download_dir=download_dir,
    )
    llm_engine = AsyncLLMEngine.from_engine_args(engine_args)

    # ---- Reranker (tiny CPU) ----
    rr_model_id = _env("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    rr_device = _env("RERANKER_DEVICE", "cpu")
    reranker_model = CrossEncoder(rr_model_id, device=rr_device)

async def generate_chat_response(
    prompt: str,
    *,
    temperature: float = 0.3,
    top_p: float = 0.85,
    max_tokens: int = 96,
    stop: Optional[List[str]] = None,
) -> str:
    if llm_engine is None:
        raise RuntimeError("LLM engine not initialized")
    params = SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop=stop or []
    )
    req_id = await llm_engine.add_request(prompt=prompt, sampling_params=params)
    out = await llm_engine.get_request_output(req_id, timeout=float(_env("VLLM_TIMEOUT", "90")))
    return out.outputs[0].text

def rerank_documents(query: str, documents: List[str], top_k: Optional[int] = None) -> List[Tuple[str, float]]:
    if reranker_model is None or not documents:
        return []
    pairs = [[query, d] for d in documents]
    scores = reranker_model.predict(pairs)
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k] if top_k else ranked

def get_llm_engine() -> AsyncLLMEngine:
    if llm_engine is None:
        raise RuntimeError("LLM engine not initialized")
    return llm_engine
