# app/main.py
import os, json, asyncio, time
from typing import AsyncIterator, Optional
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx

from app.inference import load_models, rerank_documents
from app.schemas import (
    ChatRequest, ChatResponse,
    RerankRequest, RerankResponse, RerankResponseItem
)

app = FastAPI(title="Front Office LLM (OpenAI proxy) + Mini Reranker")

# CORS (relax now; tighten in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ---- Config / State ----
DEFAULT_VLLM_PORT = int(os.getenv("VLLM_PORT", "8001"))
PROBE_RANGE = int(os.getenv("VLLM_PROBE_RANGE", "3"))  # try 8001..8001+range
OPENAI_BASE_ENV = os.getenv("LLM_OPENAI_BASE")  # force URL if set
MODEL_ID = os.getenv("LLM_MODEL", "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4")

class VLLMState:
    client: Optional[httpx.AsyncClient] = None
    base_url: Optional[str] = None

state = VLLMState()

# ---- Helpers ---------------------------------------------------------------

def _llama31_chat_template(user: str, system: Optional[str] = None) -> str:
    """
    Minimal Llama 3/3.1 chat template rendered to a single prompt string.
    Works even if the model mirror lacks a tokenizer chat_template.
    """
    sys = (system or "You are a helpful assistant.").strip()
    usr = (user or "").strip()
    # Llama-3.x special tokens
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{sys}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{usr}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

async def _probe_vllm_base() -> str:
    """Find a working OpenAI base URL. Prefer env; else probe localhost ports."""
    if OPENAI_BASE_ENV:
        base = OPENAI_BASE_ENV.rstrip("/")
        async with httpx.AsyncClient(timeout=2.0) as cx:
            try:
                r = await cx.get(f"{base}/v1/models")
                if r.status_code == 200:
                    return base
            except Exception:
                pass
    for off in range(PROBE_RANGE + 1):
        url = f"http://127.0.0.1:{DEFAULT_VLLM_PORT + off}"
        try:
            async with httpx.AsyncClient(timeout=2.0) as cx:
                r = await cx.get(f"{url}/v1/models")
                if r.status_code == 200:
                    return url
        except Exception:
            continue
    raise RuntimeError("Could not reach vLLM OpenAI server on expected ports.")

async def _ensure_client(force_refresh: bool = False) -> httpx.AsyncClient:
    """Ensure we have a connected AsyncClient with correct base_url."""
    if force_refresh and state.client:
        await state.client.aclose()
        state.client = None
        state.base_url = None
    if state.client is None or state.base_url is None:
        base = await _probe_vllm_base()
        state.client = httpx.AsyncClient(base_url=base, timeout=None)
        state.base_url = base
    return state.client

# ---- Lifecycle -------------------------------------------------------------

@app.on_event("startup")
async def _startup():
    load_models()  # CPU reranker
    # Prime vLLM client (wait until it is reachable)
    await _ensure_client(force_refresh=True)

# ---- Diagnostics -----------------------------------------------------------

@app.get("/health")
async def health():
    try:
        cx = await _ensure_client()
        r = await cx.get("/v1/models")
        ok = (r.status_code == 200)
        return {"ok": ok, "vllm_base": state.base_url, "status": r.status_code}
    except Exception as e:
        return {"ok": False, "vllm_base": state.base_url, "error": str(e)}

@app.get("/")
def root():
    return {"ok": True, "service": "frontoffice-llm-proxy"}

@app.get("/__debug_base")
async def debug_base():
    return {
        "openai_base": state.base_url or "<uninitialized>",
        "probe_from": DEFAULT_VLLM_PORT,
        "probe_range": PROBE_RANGE,
    }

# ---- Core endpoints --------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    1) Try /v1/chat/completions
    2) On failure (non-200 or {"error":"Internal Server Error"}), fall back to
       /v1/completions with a Llama-3.1 string template.
    """
    payload_chat = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": req.prompt}],
        "temperature": req.temperature,
        "top_p": req.top_p,
        "max_tokens": req.max_tokens,
        "stop": req.stop or [],
        "stream": False,
    }

    async def call_chat_once() -> httpx.Response:
        cx = await _ensure_client()
        return await cx.post("/v1/chat/completions", json=payload_chat)

    # First attempt: chat/completions
    try:
        resp = await call_chat_once()
        if resp.status_code != 200:
            # If the vLLM server changed ports, refresh & retry once
            cx = await _ensure_client(force_refresh=True)
            resp = await cx.post("/v1/chat/completions", json=payload_chat)
    except Exception as e:
        # Connection-level issue
        return JSONResponse(status_code=502, content={"error": "vLLM connection failed", "detail": str(e)})

    # If 200 OK and sane body, return it
    if resp.status_code == 200:
        try:
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return ChatResponse(text=text)
        except Exception:
            # fall through to completions fallback
            pass
    else:
        # If vLLM explicitly says Internal Server Error, fall back
        try:
            j = resp.json()
            if isinstance(j, dict) and j.get("error") == "Internal Server Error":
                pass  # trigger fallback below
            else:
                # non-200 other error: return details
                return JSONResponse(status_code=502, content={"error": "vLLM non-200", "status": resp.status_code, "body": j})
        except Exception:
            return JSONResponse(status_code=502, content={"error": "vLLM non-200", "status": resp.status_code, "raw": resp.text})

    # Fallback: /v1/completions with manual Llama-3.1 template
    prompt = _llama31_chat_template(req.prompt)
    payload_comp = {
        "model": MODEL_ID,
        "prompt": prompt,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "max_tokens": req.max_tokens,
        "stop": req.stop or [],
        "stream": False,
    }
    try:
        cx = await _ensure_client()
        r2 = await cx.post("/v1/completions", json=payload_comp)
        if r2.status_code != 200:
            # refresh base and retry once
            cx = await _ensure_client(force_refresh=True)
            r2 = await cx.post("/v1/completions", json=payload_comp)
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": "vLLM connection failed (fallback)", "detail": str(e)})

    if r2.status_code != 200:
        try:
            body = r2.json()
        except Exception:
            body = {"raw": r2.text}
        return JSONResponse(status_code=502, content={"error": "vLLM non-200 (fallback)", "status": r2.status_code, "body": body})

    data2 = r2.json()
    try:
        text2 = data2["choices"][0]["text"]
    except Exception:
        return JSONResponse(status_code=502, content={"error": "unexpected vLLM payload (fallback)", "body": data2})
    return ChatResponse(text=text2.strip())

@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest):
    ranked = rerank_documents(req.query, req.documents, top_k=req.top_k)
    return RerankResponse(results=[RerankResponseItem(document=d, score=float(s)) for d, s in ranked])

# ----- Streaming SSE: chat first, then completions fallback -----------------

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    payload_chat = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": req.prompt}],
        "temperature": req.temperature,
        "top_p": req.top_p,
        "max_tokens": req.max_tokens,
        "stop": req.stop or [],
        "stream": True,
    }

    async def stream_openai(url_path: str, json_payload: dict) -> AsyncIterator[str]:
        full = []
        started = time.time()
        cx = await _ensure_client()
        async with cx.stream("POST", url_path, json=json_payload) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                yield f'data: {json.dumps({"type":"error","message":body.decode("utf-8","ignore")})}\n\n'
                return
            yield f'data: {json.dumps({"type":"started"})}\n\n'
            async for line in resp.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:].strip()
                    if data == "[DONE]":
                        usage = {"latency_ms": int((time.time() - started) * 1000)}
                        yield f'data: {json.dumps({"type":"final","text":"".join(full),"usage":usage})}\n\n'
                        yield 'event: end\ndata: [DONE]\n\n'
                        break
                    try:
                        obj = json.loads(data)
                        delta = obj["choices"][0].get("delta", {}).get("content", "")
                    except Exception:
                        delta = ""
                    if delta:
                        full.append(delta)
                        yield f'data: {json.dumps({"type":"delta","delta":delta})}\n\n'
                if await request.is_disconnected():
                    return

    async def gen():
        # Try chat/completions first
        try:
            async for chunk in stream_openai("/v1/chat/completions", payload_chat):
                yield chunk
            return
        except Exception:
            pass

        # If that didn’t work, fall back to /v1/completions with manual template
        prompt = _llama31_chat_template(req.prompt)
        payload_comp = {
            "model": MODEL_ID,
            "prompt": prompt,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "max_tokens": req.max_tokens,
            "stop": req.stop or [],
            "stream": True,
        }
        async with (await _ensure_client()).stream("POST", "/v1/completions", json=payload_comp) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                yield f'data: {json.dumps({"type":"error","message":body.decode("utf-8","ignore")})}\n\n'
                return
            yield f'data: {json.dumps({"type":"started"})}\n\n'
            full = []
            started = time.time()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:].strip()
                    if data == "[DONE]":
                        usage = {"latency_ms": int((time.time() - started) * 1000)}
                        yield f'data: {json.dumps({"type":"final","text":"".join(full),"usage":usage})}\n\n'
                        yield 'event: end\ndata: [DONE]\n\n'
                        break
                    try:
                        obj = json.loads(data)
                        # /v1/completions streams "text" at choices[0].text
                        delta = obj["choices"][0].get("text", "")
                    except Exception:
                        delta = ""
                    if delta:
                        full.append(delta)
                        yield f'data: {json.dumps({"type":"delta","delta":delta})}\n\n'
                if await request.is_disconnected():
                    return

    headers = {"Cache-Control":"no-cache","Connection":"keep-alive","X-Accel-Buffering":"no"}
    return StreamingResponse(gen(), headers=headers, media_type="text/event-stream")
