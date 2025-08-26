# app/inference.py
from __future__ import annotations
import os
from typing import List, Optional, Tuple
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

load_dotenv()

_reranker: CrossEncoder | None = None

def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    return v if v not in (None, "") else default

def load_models() -> None:
    global _reranker
    if _reranker is not None:
        return
    rr_model_id = _env("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    rr_device = _env("RERANKER_DEVICE", "cpu")
    _reranker = CrossEncoder(rr_model_id, device=rr_device)

def rerank_documents(query: str, documents: List[str], top_k: Optional[int] = None) -> List[Tuple[str, float]]:
    if _reranker is None or not documents:
        return []
    pairs = [[query, d] for d in documents]
    scores = _reranker.predict(pairs)
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k] if top_k else ranked
