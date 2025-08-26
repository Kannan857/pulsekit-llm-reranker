# app/inference.py
import os
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------- Config --------
RERANKER_MODEL_ID = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "cpu").lower()  # "cpu" or "cuda"
_MAX_LEN = int(os.getenv("RERANKER_MAX_LEN", "512"))

# -------- Globals --------
_tokenizer = None
_model = None
_device = torch.device("cuda" if (RERANKER_DEVICE.startswith("cuda") and torch.cuda.is_available()) else "cpu")


def load_models() -> None:
    """
    Load the cross-encoder reranker using vanilla Transformers
    (no sentence-transformers, no sklearn).
    """
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return

    _tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_ID)
    _model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_ID)
    _model.to(_device)
    _model.eval()


@torch.inference_mode()
def _score_pairs(queries: List[str], docs: List[str]) -> torch.Tensor:
    """
    Score (query, doc) pairs with the loaded cross-encoder.
    Returns a tensor of shape [N] with higher = better.
    """
    assert _tokenizer is not None and _model is not None, "Call load_models() first."

    enc = _tokenizer(
        queries,
        docs,
        padding=True,
        truncation=True,
        max_length=_MAX_LEN,
        return_tensors="pt",
    )
    enc = {k: v.to(_device) for k, v in enc.items()}
    logits = _model(**enc).logits  # shape [N, 1] (regression) OR [N, 2] (classification)

    if logits.dim() == 1:
        scores = logits
    elif logits.size(-1) == 1:
        scores = logits.squeeze(-1)
    else:
        # assume binary classification; use prob of positive class
        scores = torch.softmax(logits, dim=-1)[:, 1]

    return scores.detach().float().cpu()


def rerank_documents(query: str, documents: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Returns top_k documents with scores, sorted descending.
    """
    if not documents:
        return []

    if _tokenizer is None or _model is None:
        load_models()

    q_list = [query] * len(documents)
    scores = _score_pairs(q_list, documents).tolist()

    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    if top_k and top_k > 0:
        ranked = ranked[:top_k]
    return ranked
