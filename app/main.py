# app/main.py
import os
import json
import time
import asyncio
from typing import Optional, AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from app.inference import load_models, rerank_documents
from app.schemas import (
    ChatRequest, ChatResponse,
    RerankRequest, RerankResponse, RerankResponseItem,
)

app = FastAPI(title="Front Office LLM (OpenAI proxy) + Mini Reranker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

DEFAULT_VLLM_PORT = int(os.getenv("VLLM_PORT", "8001"))
PROBE_RANGE = int(os.getenv("VLLM_PROBE_RANGE", "3"))  # try 8001..+range
OPENAI_BASE_ENV = os.getenv("LLM_OPENAI_BASE")  # optional override
MODEL_ID = os.getenv("LLM_MODEL", "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4")

class VLLMState:
    client: Optional[httpx.AsyncClient] = None
    base_url: Optional[str] = None

state = VLLMState()

def _llama31_chat_template(user: str, system: Optional[str] = None) -> str:
    sys = (system or "You are a helpful assistant.").strip()
    usr = (user or "").strip()
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{sys}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{usr}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

async def _probe_vllm_base() -> str:
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
    if force_refresh and state.client:
        try: await state.client.aclose()
        except Exception: pass
        state.client = None
        state.base_url = None
    if state.client is None or state.base_url is None:
        base = await _probe_vllm_base()
        state.client = httpx.AsyncClient(base_url=base, timeout=None)
        state.base_url = base
    return state.client

@app.on_event("startup")
async def _startup():
    load_models()
    async def _bg_warm():
        for _ in range(180):
            try:
                await _ensure_client(force_refresh=True)
                return
            except Exception:
                await asyncio.sleep(1)
    asyncio.create_task(_bg_warm())

@app.on_event("shutdown")
async def _shutdown():
    if state.client:
        try: await state.client.aclose()
        except Exception: pass

@app.get("/health")
async def health():
    try:
        cx = await _ensure_client()
        r = await cx.get("/v1/models")
        return {"ok": r.status_code == 200, "vllm_base": state.base_url, "status": r.status_code}
    except Exception as e:
        return {"ok": False, "vllm_base": state.base_url, "error": str(e)}

@app.get("/")
def root():
    return {"ok": True, "service": "frontoffice-llm-proxy"}

@app.get("/__debug_base")
async def debug_base():
    return {"openai_base": state.base_url or "<uninitialized>", "probe_from": DEFAULT_VLLM_PORT, "probe_range": PROBE_RANGE}

# ----- /chat with unconditional fallback to /v1/completions -----
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    payload_chat = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": req.prompt}],
        "temperature": req.temperature,
        "top_p": req.top_p,
        "max_tokens": req.max_tokens,
        "stop": req.stop or [],
        "stream": False,
    }

    async def post_chat_once():
        cx = await _ensure_client()
        return await cx.post("/v1/chat/completions", json=payload_chat)

    # Try chat/completions (2 attempts, re-detect base on failure)
    try:
        resp = await post_chat_once()
        if resp.status_code != 200:
            await _ensure_client(force_refresh=True)
            resp = await post_chat_once()
    except Exception as e:
        # connection failure -> go to fallback path below
        resp = None
        err_detail = str(e)
    else:
        err_detail = None

    # If chat/completions succeeded and body parsed, return it
    if resp is not None and resp.status_code == 200:
        try:
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return ChatResponse(text=text)
        except Exception:
            pass  # fall through to fallback

    # Fallback: /v1/completions with manual template (ALWAYS for non-200 or parse error)
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
            await _ensure_client(force_refresh=True)
            r2 = await cx.post("/v1/completions", json=payload_comp)
    except Exception as e:
        return JSONResponse(status_code=502, content={
            "error": "vLLM connection failed (fallback)",
            "detail": str(e),
            "prev_chat_error": err_detail,
        })

    if r2.status_code != 200:
        try: body = r2.json()
        except Exception: body = {"raw": r2.text}
        return JSONResponse(status_code=502, content={
            "error": "vLLM non-200 (fallback)",
            "status": r2.status_code,
            "body": body,
            "prev_chat_error": err_detail,
        })

    data2 = r2.json()
    try:
        text2 = data2["choices"][0]["text"]
    except Exception:
        return JSONResponse(status_code=502, content={
            "error": "unexpected vLLM payload (fallback)",
            "body": data2,
            "prev_chat_error": err_detail,
        })
    return ChatResponse(text=text2.strip())

# ----- rerank -----
@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest):
    ranked = rerank_documents(req.query, req.documents, top_k=req.top_k)
    return RerankResponse(results=[RerankResponseItem(document=d, score=float(s)) for d, s in ranked])

# ----- streaming: try chat, then completions if non-200 -----
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

    async def stream_openai(path: str, payload: dict) -> AsyncIterator[str]:
        full = []
        started = time.time()
        cx = await _ensure_client()
        async with cx.stream("POST", path, json=payload) as resp:
            if resp.status_code != 200:
                # indicate non-200; caller will try fallback
                yield "__NON_200__"
                return
            yield f'data: {json.dumps({"type":"started"})}\n\n'
            async for line in resp.aiter_lines():
                if not line: continue
                if line.startswith("data: "):
                    data = line[6:].strip()
                    if data == "[DONE]":
                        usage = {"latency_ms": int((time.time()-started)*1000)}
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
        # first try chat/completions
        non200 = False
        async for chunk in stream_openai("/v1/chat/completions", payload_chat):
            if chunk == "__NON_200__":
                non200 = True
                break
            yield chunk
        if not non200:
            return
        # fallback to /v1/completions with template
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
        cx = await _ensure_client()
        async with cx.stream("POST", "/v1/completions", json=payload_comp) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                yield f'data: {json.dumps({"type":"error","message":body.decode("utf-8","ignore")})}\n\n'
                return
            yield f'data: {json.dumps({"type":"started"})}\n\n'
            full = []
            started = time.time()
            async for line in resp.aiter_lines():
                if not line: continue
                if line.startswith("data: "):
                    data = line[6:].strip()
                    if data == "[DONE]":
                        usage = {"latency_ms": int((time.time()-started)*1000)}
                        yield f'data: {json.dumps({"type":"final","text":"".join(full),"usage":usage})}\n\n'
                        yield 'event: end\ndata: [DONE]\n\n'
                        break
                    try:
                        obj = json.loads(data)
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
