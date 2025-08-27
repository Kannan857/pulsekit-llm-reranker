import os
import json
import time
import asyncio
from typing import Optional, AsyncIterator, Dict, Any

import anyio
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from app.inference import load_models, rerank_documents
from app.schemas import (
    ChatRequest, ChatResponse,
    RerankRequest, RerankResponse, RerankResponseItem,
    GenerationParams, Message
)

# ------------------------------
# App & CORS
# ------------------------------
app = FastAPI(title="PulseKit LLM Gateway (vLLM proxy + Reranker)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# Environment & defaults
# ------------------------------
DEFAULT_VLLM_PORT = int(os.getenv("VLLM_PORT", "8001"))
VLLM_BASE = os.getenv("LLM_OPENAI_BASE", f"http://127.0.0.1:{DEFAULT_VLLM_PORT}").rstrip("/")
MODEL_ID = os.getenv("LLM_MODEL", "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4")

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are PulseKit’s helpful assistant. Be concise and helpful. If unsure, ask a clarifying question."
)

# Server-side fallback defaults (used only if caller doesn't pass req.params or req.defaults)
VOICE_MAX_TOKENS_DEFAULT = int(os.getenv("VOICE_MAX_TOKENS_DEFAULT", "96"))
CHAT_MAX_TOKENS_DEFAULT  = int(os.getenv("CHAT_MAX_TOKENS_DEFAULT", "256"))
VOICE_TEMPERATURE_DEFAULT = float(os.getenv("VOICE_TEMPERATURE_DEFAULT", "0.25"))
CHAT_TEMPERATURE_DEFAULT  = float(os.getenv("CHAT_TEMPERATURE_DEFAULT", "0.35"))
VOICE_TOPP_DEFAULT = float(os.getenv("VOICE_TOPP_DEFAULT", "0.9"))
CHAT_TOPP_DEFAULT  = float(os.getenv("CHAT_TOPP_DEFAULT", "0.9"))
VOICE_HISTORY_TURNS = int(os.getenv("VOICE_HISTORY_TURNS", "3"))
CHAT_HISTORY_TURNS  = int(os.getenv("CHAT_HISTORY_TURNS", "8"))

DEFAULT_STOPS_ENV = os.getenv("LLM_STOPS", "")  # comma-separated
DEFAULT_STOPS = (
    [s for s in (x.strip() for x in DEFAULT_STOPS_ENV.split(",")) if s]
    if DEFAULT_STOPS_ENV else [".\n", "\n\n", "</s>"]
)

# ------------------------------
# vLLM client state (single pooled client)
# ------------------------------
class VLLMState:
    client: Optional[httpx.AsyncClient] = None
    base_url: str = VLLM_BASE

state = VLLMState()

def _ensure_client_created() -> httpx.AsyncClient:
    if state.client is None:
        limits = httpx.Limits(max_connections=200, max_keepalive_connections=100)
        timeout = httpx.Timeout(connect=2.0, read=None, write=None, pool=None)
        state.client = httpx.AsyncClient(base_url=state.base_url, limits=limits, timeout=timeout)
    return state.client

# ------------------------------
# Helpers: apply channel defaults with caller-supplied defaults
# Precedence: req.params > req.defaults[channel] > server defaults
# ------------------------------
def _pick(val, *fallbacks):
    for v in (val, *fallbacks):
        if v is not None:
            return v
    return None

def _resolve_gen_params(channel: str, req: ChatRequest) -> GenerationParams:
    # Caller-supplied per-channel defaults (middle precedence)
    chan_defaults: Optional[GenerationParams] = None
    if req.defaults:
        if channel == "voice":
            chan_defaults = req.defaults.voice
        else:
            chan_defaults = req.defaults.chat

    # Server defaults (lowest precedence)
    if channel == "voice":
        server_defaults = GenerationParams(
            temperature=VOICE_TEMPERATURE_DEFAULT,
            top_p=VOICE_TOPP_DEFAULT,
            max_tokens=VOICE_MAX_TOKENS_DEFAULT,
            stop=DEFAULT_STOPS,
        )
    else:
        server_defaults = GenerationParams(
            temperature=CHAT_TEMPERATURE_DEFAULT,
            top_p=CHAT_TOPP_DEFAULT,
            max_tokens=CHAT_MAX_TOKENS_DEFAULT,
            stop=DEFAULT_STOPS,
        )

    # Highest precedence: per-request overrides
    p = req.params or GenerationParams()

    return GenerationParams(
        temperature=_pick(p.temperature, getattr(chan_defaults, "temperature", None), server_defaults.temperature),
        top_p=_pick(p.top_p, getattr(chan_defaults, "top_p", None), server_defaults.top_p),
        max_tokens=_pick(p.max_tokens, getattr(chan_defaults, "max_tokens", None), server_defaults.max_tokens),
        stop=_pick(p.stop, getattr(chan_defaults, "stop", None), server_defaults.stop),
    )

def _trim_history(channel: str, history: Optional[list[Message]]) -> list[Message]:
    history = history or []
    n = VOICE_HISTORY_TURNS if channel == "voice" else CHAT_HISTORY_TURNS
    return history[-n:] if n > 0 else []

def _build_messages(channel: str, req: ChatRequest) -> list[dict]:
    msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if req.context:
        msgs.append({"role": "system", "content": f"Context (use only if relevant):\n{req.context}"})
    msgs.extend([m.model_dump() for m in _trim_history(channel, req.history)])
    msgs.append({"role": "user", "content": req.prompt})
    return msgs

def _payload_for_vllm(req: ChatRequest) -> Dict[str, Any]:
    ch = req.channel or "chat"
    gen = _resolve_gen_params(ch, req)
    messages = _build_messages(ch, req)
    return {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": gen.temperature,
        "top_p": gen.top_p,
        "max_tokens": gen.max_tokens,
        "stop": gen.stop or DEFAULT_STOPS,
        "user": req.user or "anon",
    }

# ------------------------------
# Lifecycle: real warm-up (1-token gen)
# ------------------------------
@app.on_event("startup")
async def _startup():
    # load local CPU models (e.g., reranker)
    load_models()
    _ensure_client_created()

    async def _bg_warm():
        try:
            await state.client.get("/v1/models")
            payload = {
                "model": MODEL_ID,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "hi"},
                ],
                "max_tokens": 1,
                "stream": False,
            }
            await state.client.post("/v1/chat/completions", json=payload)
        except Exception:
            pass

    asyncio.create_task(_bg_warm())

@app.on_event("shutdown")
async def _shutdown():
    if state.client:
        try:
            await state.client.aclose()
        except Exception:
            pass

# ------------------------------
# Health & debug
# ------------------------------
@app.get("/health")
async def health():
    try:
        _ensure_client_created()
        r = await state.client.get("/v1/models")
        return {"ok": r.status_code == 200, "vllm_base": state.base_url, "status": r.status_code}
    except Exception as e:
        return {"ok": False, "vllm_base": state.base_url, "error": str(e)}

@app.get("/")
def root():
    return {"ok": True, "service": "pulsekit-llm-gateway"}

@app.get("/__debug_base")
def debug_base():
    return {"openai_base": state.base_url}

# ------------------------------
# Chat (non-stream) — single path to /v1/chat/completions
# ------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        payload = _payload_for_vllm(req) | {"stream": False}
        _ensure_client_created()
        r = await state.client.post("/v1/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        return ChatResponse(text=text)
    except httpx.ConnectError as e:
        return JSONResponse(status_code=502, content={"error": "connect", "detail": str(e)})
    except httpx.HTTPStatusError as e:
        return JSONResponse(
            status_code=502,
            content={"error": "upstream", "status": e.response.status_code, "body": e.response.text},
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "unexpected", "detail": str(e)})

# ------------------------------
# Chat (stream) — minimal-pass-through SSE
# ------------------------------
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    payload = _payload_for_vllm(req) | {"stream": True}

    async def gen() -> AsyncIterator[str]:
        try:
            _ensure_client_created()
            async with state.client.stream("POST", "/v1/chat/completions", json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    yield "event: error\n"
                    yield f"data: {json.dumps({'status': resp.status_code, 'body': body.decode('utf-8','ignore')})}\n\n"
                    return
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    # vLLM already yields OpenAI-style 'data: {...}' lines → pass through
                    yield f"{line}\n"
                    if await request.is_disconnected():
                        return
        except httpx.ConnectError as e:
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"

    headers = {"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    return StreamingResponse(gen(), headers=headers, media_type="text/event-stream")

# ------------------------------
# Rerank — offload to thread; keep GPU free
# ------------------------------
@app.post("/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest):
    try:
        ranked = await anyio.to_thread.run_sync(
            rerank_documents, req.query, req.documents, req.top_k
        )
        return RerankResponse(results=[RerankResponseItem(document=d, score=float(s)) for d, s in ranked])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"rerank error: {e}")
