# app/main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager

from . import models, inference

# --- App Lifecycle ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup and shutdown events."""
    # On startup, load the models into memory
    inference.load_models()
    yield
    # On shutdown, you can add cleanup logic here if needed

app = FastAPI(lifespan=lifespan)


# --- API Endpoints ---
@app.get("/health")
async def health_check():
    """Health check endpoint to verify the service is running."""
    return {"status": "ok", "model_ready": True}

@app.post("/chat", response_model=models.ChatResponse)
async def chat_endpoint(request: models.ChatRequest):
    """Endpoint for chat interactions with the LLM."""
    response_text = await inference.generate_chat_response(request.prompt)
    return models.ChatResponse(response=response_text)

@app.post("/rerank", response_model=models.RerankResponse)
async def rerank_endpoint(request: models.RerankRequest):
    """Endpoint for reranking documents."""
    ranked_docs = inference.rerank_documents(request.query, request.documents)
    return models.RerankResponse(ranked_documents=ranked_docs)