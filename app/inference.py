# app/inference.py
from __future__ import annotations

import os
import uuid
import inspect
from typing import List, Optional, Tuple, Any

import torch
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams

load_dotenv()

llm_engine: Optional[AsyncLLMEngine] = None
reranker_model: Optional[CrossEncoder] = None


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v not in (None, "") else default


def load_models() -> None:
    """Initialize vLLM (optionally quantized) + lightweight CPU reranker."""
    global llm_engine, reranker_model

    # ---- LLM via vLLM ----
    model_id = _env("LLM_MODEL", "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4")
    quantization = _env("LLM_QUANTIZATION", "awq")  # set "" for FP model
    tp = int(_env("VLLM_TP", "1"))
    max_ctx = int(_env("VLLM_MAX_LEN", "8192"))
    gpu_util = float(_env("VLLM_GPU_UTIL", "0.90"))
    download_dir = _env("HF_HOME") or _env("TRANSFORMERS_CACHE")

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


def _make_add_request_caller():
    """
    Detect and return a function that calls llm_engine.add_request with the
    right signature and returns (collector, request_id).
    """
    if llm_engine is None:
        raise RuntimeError("LLM engine not initialized")

    sig = inspect.signature(llm_engine.add_request)
    names = list(sig.parameters.keys())

    # Variant A: request_id is required (your build)
    if "request_id" in names:
        if "params" in names:
            async def caller(prompt, params):
                rid = uuid.uuid4().hex
                collector = await llm_engine.add_request(
                    request_id=rid, prompt=prompt, params=params
                )
                return collector, rid
            return caller
        elif "sampling_params" in names:
            async def caller(prompt, params):
                rid = uuid.uuid4().hex
                collector = await llm_engine.add_request(
                    request_id=rid, prompt=prompt, sampling_params=params
                )
                return collector, rid
            return caller
        else:
            async def caller(prompt, params):
                rid = uuid.uuid4().hex
                collector = await llm_engine.add_request(rid, prompt, params)
                return collector, rid
            return caller

    # Variant B: no request_id, keyword 'params'
    if "params" in names:
        async def caller(prompt, params):
            collector = await llm_engine.add_request(prompt=prompt, params=params)
            rid = getattr(collector, "request_id", uuid.uuid4().hex)
            return collector, rid
        return caller

    # Variant C: no request_id, legacy 'sampling_params'
    if "sampling_params" in names:
        async def caller(prompt, params):
            collector = await llm_engine.add_request(prompt=prompt, sampling_params=params)
            rid = getattr(collector, "request_id", uuid.uuid4().hex)
            return collector, rid
        return caller

    # Variant D: positional only (prompt, params)
    async def caller(prompt, params):
        collector = await llm_engine.add_request(prompt, params)
        rid = getattr(collector, "request_id", uuid.uuid4().hex)
        return collector, rid

    return caller


async def _collector_to_final_text(collector: Any) -> str:
    """
    Consume a RequestOutputCollector and return the final text.
    Works whether the collector is async-iterable or exposes a get_final_output().
    """
    # Prefer async iteration (most V1 collectors support this)
    try:
        last = None
        async for out in collector:
            last = out
        if last is not None:
            return last.outputs[0].text
    except TypeError:
        # Not async-iterable; fall back to method calls
        pass

    # Try get_final_output()
    if hasattr(collector, "get_final_output"):
        out = await collector.get_final_output()
        return out.outputs[0].text

    # Try wait() or result() patterns (very defensive)
    if hasattr(collector, "wait"):
        out = await collector.wait()
        return out.outputs[0].text
    if hasattr(collector, "result"):
        out = await collector.result()
        return out.outputs[0].text

    # As a last resort, raise a helpful error
    raise RuntimeError("Unknown collector interface; cannot obtain final output.")


async def generate_chat_response(
    prompt: str,
    *,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_tokens: int = 256,
    stop: Optional[List[str]] = None,
) -> str:
    """One-shot generation with per-request sampling controls."""
    if llm_engine is None:
        raise RuntimeError("LLM engine not initialized")

    params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=stop or [],
    )

    add_request = _make_add_request_caller()
    collector, _ = await add_request(prompt, params)

    # Consume collector to final text (no get_request_output in V1)
    return await _collector_to_final_text(collector)


def rerank_documents(
    query: str,
    documents: List[str],
    top_k: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """Lightweight cross-encoder reranking (CPU by default)."""
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


def get_add_request_signature_str() -> str:
    if llm_engine is None:
        return "<engine not initialized>"
    return str(inspect.signature(llm_engine.add_request))
