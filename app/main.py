# app/main.py
import os, json, asyncio, time
from typing import AsyncIterator
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx

from app.inference import load_models, rerank_documents
from app.schemas import (
    ChatRequest, ChatResponse,
    RerankRequest, RerankResponse, RerankResponseItem
)

app = FastAPI(title="Front Office LLM (Quant via OpenAI API) + Mini Reranker")

# CORS (relax for now; tighten in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

VLLM_PORT = int(os.getenv("VLLM_PORT", "8001"))
OPENAI_BASE = os.getenv("LLM_OPENAI_BASE", f"http://127.0.0.1:{VLLM_PORT}")
OPENAI_CHAT_URL = f"{OPENAI_BASE}/v1/chat/completions"
MODEL_ID = os.getenv("LLM_MODEL", "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4")

@app.on_event("startup")
async def _startup():
    load_models()
    # optional: wait for vLLM to come up
    async with httpx.AsyncClient(timeout=2.5) as cx:
        for _ in range(60):
            try:
                r = await cx.get(f"{OPENAI_BASE}/v1/models")
                if r.status_code == 200:
                    break
            except Exception:
                pass
            await asyncio.sleep(1)

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/")
def root():
    return {"ok": True, "service": "frontoffice-llm-proxy"}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": req.prompt}],
        "temperature": req.temperature,
        "top_p": req.top_p,
        "max_tokens": req.max_tokens,
        "stop": req.stop or [],
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=None) as cx:
        resp = await cx.post(OPENAI_CHAT_URL, json=payload)
        if resp.status_code != 200:
            try:
                err = resp.json()
            except Exception:
                err = {"error": resp.text}
            raise RuntimeError(f"vLLM error: {err}")
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return ChatResponse(text=text)

@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest):
    ranked = rerank_documents(req.query, req.documents, top_k=req.top_k)
    return RerankResponse(results=[RerankResponseItem(document=d, score=float(s)) for d, s in ranked])

# ----- Streaming SSE: transform OpenAI chunks -> {type:delta, delta:"..."} -----
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": req.prompt}],
        "temperature": req.temperature,
        "top_p": req.top_p,
        "max_tokens": req.max_tokens,
        "stop": req.stop or [],
        "stream": True,
    }

    async def event_gen() -> AsyncIterator[str]:
        full = []
        started = time.time()
        async with httpx.AsyncClient(timeout=None) as cx:
            async with cx.stream("POST", OPENAI_CHAT_URL, json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    yield f'data: {json.dumps({"type":"error","message":body.decode("utf-8","ignore")})}\n\n'
                    return

                # Emit a simple started event (no request_id from server)
                yield f'data: {json.dumps({"type":"started"})}\n\n'

                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data = line[6:].strip()
                        if data == "[DONE]":
                            # Final event
                            usage = {"latency_ms": int((time.time() - started) * 1000)}
                            yield f'data: {json.dumps({"type":"final","text":"".join(full),"usage":usage})}\n\n'
                            yield 'event: end\ndata: [DONE]\n\n'
                            break
                        try:
                            obj = json.loads(data)
                        except Exception:
                            continue
                        # OpenAI chunk -> delta text
                        try:
                            delta = obj["choices"][0]["delta"].get("content", "")
                        except Exception:
                            delta = ""
                        if delta:
                            full.append(delta)
                            yield f'data: {json.dumps({"type":"delta","delta":delta})}\n\n'
                    # optional: cancel on client disconnect
                    if await request.is_disconnected():
                        return

    headers = {"Cache-Control":"no-cache","Connection":"keep-alive","X-Accel-Buffering":"no"}
    return StreamingResponse(event_gen(), headers=headers, media_type="text/event-stream")
