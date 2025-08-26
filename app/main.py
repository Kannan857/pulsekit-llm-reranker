import asyncio, json, time
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from app.inference import load_models, generate_chat_response, rerank_documents, get_llm_engine
from app.schemas import ChatRequest, ChatResponse, RerankRequest, RerankResponse, RerankResponseItem
from vllm.sampling_params import SamplingParams

app = FastAPI(title="Front Office LLM (Quant) + Mini Reranker")

@app.on_event("startup")
async def _startup():
    load_models()

@app.get("/health")
def health():
    return {"ok": True}

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
    return RerankResponse(results=[RerankResponseItem(document=d, score=float(s)) for d, s in ranked])

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    engine = get_llm_engine()
    params = SamplingParams(
        temperature=req.temperature, top_p=req.top_p, max_tokens=req.max_tokens, stop=req.stop or []
    )

    async def gen():
        started = time.time()
        req_id = await engine.add_request(prompt=req.prompt, sampling_params=params)
        yield f'data: {json.dumps({"type":"started","request_id":req_id})}\n\n'
        prev = ""
        last = None
        try:
            async for out in engine.get_generator(req_id):
                last = out
                text = out.outputs[0].text
                delta = text[len(prev):]
                prev = text
                if delta:
                    yield f'data: {json.dumps({"type":"delta","delta":delta})}\n\n'
                if await request.is_disconnected():
                    break
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

    headers = {"Cache-Control":"no-cache","Connection":"keep-alive","X-Accel-Buffering":"no"}
    return StreamingResponse(gen(), headers=headers, media_type="text/event-stream")
