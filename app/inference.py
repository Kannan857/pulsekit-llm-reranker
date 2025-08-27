# app/inference.py
import os
from typing import List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------- Config ----------------
RERANKER_MODEL_ID = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "cpu").lower()  # "cpu", "cuda", or "cuda:0"
# Keep shorter for latency; raise only if you truly need longer passages
_MAX_LEN = int(os.getenv("RERANKER_MAX_LEN", "256"))
# Prevent CPU thread storms (important for FastAPI latency)
_TORCH_THREADS = int(os.getenv("RERANKER_THREADS", "4"))
# Score in batches to avoid spikes with large doc lists
_BATCH = int(os.getenv("RERANKER_BATCH", "32"))

# Optional ONNXRuntime backend (recommended if you have a pre-exported model dir)
# Set RERANKER_ONNX_DIR=/path/to/exported/onnx/dir to enable
_RERANKER_ONNX_DIR: Optional[str] = os.getenv("RERANKER_ONNX_DIR") or None

# ---------------- Globals ----------------
_tokenizer = None
_model = None
_backend = "hf"  # "hf" or "onnx"

# Device: default CPU; allow CUDA only if explicitly requested AND available
_device = torch.device(
    RERANKER_DEVICE if (RERANKER_DEVICE.startswith("cuda") and torch.cuda.is_available())
    else "cpu"
)

# Threading caps (helps ensure the reranker doesn't hog all cores)
try:
    torch.set_num_threads(_TORCH_THREADS)
    torch.set_num_interop_threads(1)
except Exception:
    pass  # older torch versions may not support these

os.environ.setdefault("OMP_NUM_THREADS", str(_TORCH_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_TORCH_THREADS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(_TORCH_THREADS))


def load_models() -> None:
    """
    Load the cross-encoder reranker (either HF or ONNXRuntime if provided).
    """
    global _tokenizer, _model, _backend
    if _tokenizer is not None and _model is not None:
        return

    _tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_ID)

    if _RERANKER_ONNX_DIR:
        # Use ONNXRuntime if caller provided an exported dir
        try:
            from optimum.onnxruntime import ORTModelForSequenceClassification
            _model = ORTModelForSequenceClassification.from_pretrained(
                _RERANKER_ONNX_DIR, provider="CPUExecutionProvider"
            )
            _backend = "onnx"
            return
        except Exception:
            # Fallback to HF if ONNX is unavailable/misconfigured
            _backend = "hf"

    # HF (PyTorch) backend
    _model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_ID)
    _model.to(_device)
    _model.eval()
    _backend = "hf"


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
        padding=True,           # dynamic padding to the longest item in this batch
        truncation=True,
        max_length=_MAX_LEN,
        return_tensors="pt",
    )

    # ONNXRuntime models run on CPU; HF may be CPU or CUDA depending on _device
    if _backend == "hf":
        enc = {k: v.to(_device) for k, v in enc.items()}
        logits = _model(**enc).logits  # [N,1] or [N,2]
    else:
        # ORT accepts torch tensors on CPU fine
        logits = _model(**enc).logits  # [N,1] or [N,2]

    if logits.dim() == 1:
        scores = logits
    elif logits.size(-1) == 1:
        scores = logits.squeeze(-1)
    else:
        # binary classification head: take prob of positive class
        scores = torch.softmax(logits, dim=-1)[:, 1]

    # Always return on CPU
    return scores.detach().float().cpu()


def rerank_documents(query: str, documents: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Returns top_k documents with scores, sorted descending.
    """
    if not documents:
        return []

    if _tokenizer is None or _model is None:
        load_models()

    scores_all: List[float] = []
    # Batch to keep latency predictable for larger candidate sets
    for i in range(0, len(documents), _BATCH):
        chunk = documents[i : i + _BATCH]
        q_list = [query] * len(chunk)
        scores_chunk = _score_pairs(q_list, chunk).tolist()
        scores_all.extend(scores_chunk)

    ranked = sorted(zip(documents, scores_all), key=lambda x: x[1], reverse=True)
    if top_k and top_k > 0:
        ranked = ranked[:top_k]
    return ranked
