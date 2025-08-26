# app/main.py
import json
import time
import inspect
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from app.inference import (
    load_models,
    generate_chat_response,
    rerank_documents,
    get_llm_engine,
    get_add_request_signature_str,
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

# CORS (relax for now; tighten in prod)
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

@app.get("/")
def root():
    return {"ok": True, "service": "frontoffice-llm"}

@app.get("/__debug_sig")
def debug_sig():
    return {"add_request_signature": get_add_request_signature_str()}

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

# ---------- Streaming SSE endpoint (use get_generator(request_id)) ----------
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    engine = get_llm_engine()
    params = SamplingParams(
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
        stop=req.stop or [],
    )

    # Build-compatible add_request -> request_id
    sig = inspect.signature(engine.add_request)
    names = list(sig.parameters.keys())

    def enqueue(prompt, params) -> str:
        if "request_id" in names:
            rid = uuid.uuid4().hex
            if "params" in names:
                engine.add_request(request_id=rid, prompt=prompt, params=params)
            elif "sampling_params" in names:
                engine.add_request(request_id=rid, prompt=prompt, sampling_params=params)
            else:
                engine.add_request(rid, prompt, params)
            return rid
        elif "params" in names:
            engine.add_request(prompt=prompt, params=params)
            return uuid.uuid4().hex
        elif "sampling_params" in names:
            engine.add_request(prompt=prompt, sampling_params=params)
            return uuid.uuid4().hex
        else:
            engine.add_request(prompt, params)
            return uuid_uuid4().hex  # typofix below

    # (tiny typo safety)
    def uuid_uuid4():  # just in case of previous branch
        return uuid.uuid4().hex

    request_id = enqueue(req.prompt, params)

    async def gen():
        started = time.time()
        # Emit started with the request_id we generated
        yield f'data: {json.dumps({"type":"started","request_id":request_id})}\n\n'

        prev = ""
        last = None
        try:
            # Stream from engine.get_generator(request_id) — portable across V1 builds
            async for out in engine.get_generator(request_id):
                last = out
                text_so_far = out.outputs[0].text
                delta = text_so_far[len(prev):]
                if delta:
                    prev = text_so_far
                    yield f'data: {json.dumps({"type":"delta","delta":delta})}\n\n'

                # client closed connection (barge-in)
                if await request.is_disconnected():
                    try:
                        await engine.abort_request(request_id)
                    except Exception:
                        pass
                    return

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
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(gen(), headers=headers, media_type="text/event-stream")
