# app/main.py
import json
import time
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from app.inference import (
    load_models,
    generate_chat_response,
    rerank_documents,
    get_llm_engine,
)
from app.schemas import (
    ChatRequest,
    ChatResponse,
    RerankRequest,
    RerankResponse,
    RerankResponseItem,
)
from vllm.sampling_params import SamplingParams

app = FastAPI(title="Front Office LLM (Quant) + Mini Reranker")

# --- CORS (adjust allow_origins to your domains in prod) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def _startup():
    load_models()

@app.get("/health")
def health():
    return {"ok": True}

# Helpful root for platforms that probe "/"
@app.get("/")
def root():
    return {"ok": True, "service": "frontoffice-llm"}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    text = await generate_chat_response(
        prompt=req.prompt,
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
        stop=req.stop,
    )
    return ChatResponse(text=text)

@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest):
    ranked = rerank_documents(req.query, req.documents, top_k=req.top_k)
    return RerankResponse(
        results=[RerankResponseItem(document=d, score=float(s)) for d, s in ranked]
    )

# ---------- Streaming SSE endpoint (for TTS/barge-in) ----------
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    engine = get_llm_engine()
    params = SamplingParams(
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
        stop=req.stop or [],
    )

    async def gen():
        started = time.time()

        # vLLM V1 prefers positional (prompt, params). Older engines used sampling_params=...
        try:
            req_id = await engine.add_request(req.prompt, params)  # V1 style
        except TypeError:
            req_id = await engine.add_request(prompt=req.prompt, sampling_params=params)  # legacy

        # Emit a started event (helps clients correlate/cancel)
        yield f'data: {json.dumps({"type":"started","request_id":req_id})}\n\n'

        prev = ""
        last = None
        try:
            async for out in engine.get_generator(req_id):
                last = out
                text_so_far = out.outputs[0].text
                delta = text_so_far[len(prev):]
                if delta:
                    prev = text_so_far
                    yield f'data: {json.dumps({"type":"delta","delta":delta})}\n\n'

                # client closed connection (barge-in) -> cancel request
                if await request.is_disconnected():
                    try:
                        await engine.abort_request(req_id)
                    except Exception:
                        pass
                    return

            # Final usage (if available)
            usage = {}
            if last and getattr(last, "metrics", None):
                m = last.metrics
                usage = {
                    "prompt_tokens": int(getattr(m, "num_input_tokens", 0)),
                    "completion_tokens": int(getattr(m, "num_generated_tokens", 0)),
                    "latency_ms": int((time.time() - started) * 1000),
                }

            yield f'data: {json.dumps({"type":"final","text":prev,"usage":usage})}\n\n'
            yield 'event: end\ndata: [DONE]\n\n'

        except Exception as e:
            yield f'data: {json.dumps({"type":"error","message":str(e)})}\n\n'

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # disable proxy buffering if present
    }
    return StreamingResponse(gen(), headers=headers, media_type="text/event-stream")
