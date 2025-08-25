# app/inference.py
import os
import torch
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

load_dotenv()

# --- Global Model Objects ---
llm_engine = None
reranker_model = None

# --- Model Loading ---
def load_models():
    """
    Loads the LLM and reranker models into memory.
    This is called once during the application startup.
    """
    global llm_engine, reranker_model
    
    llm_model_name = os.getenv("LLM_MODEL")
    reranker_model_name = os.getenv("RERANKER_MODEL")
    
    print("Loading models...")
    
    # 1. Load vLLM Engine
    # Using AsyncEngineArgs allows for non-blocking requests
    engine_args = AsyncEngineArgs(
        model=llm_model_name,
        quantization='awq',
        dtype='auto',
        tensor_parallel_size=1, # Adjust if using multiple GPUs
        gpu_memory_utilization=0.90
    )
    llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    # 2. Load Reranker Model
    reranker_model = CrossEncoder(
        reranker_model_name,
        max_length=512,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print("Models loaded successfully.")

# --- Inference Functions ---
async def generate_chat_response(prompt: str):
    """Generates a response from the LLM."""
    if not llm_engine:
        raise RuntimeError("LLM engine is not initialized.")

    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=512)
    request_id = f"chat-{os.urandom(12).hex()}"
    
    # vLLM's async generate method
    results_generator = llm_engine.generate(prompt, sampling_params, request_id)
    
    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    return final_output.outputs[0].text


def rerank_documents(query: str, documents: list):
    """Reranks a list of documents based on a query."""
    if not reranker_model:
        raise RuntimeError("Reranker model is not initialized.")
    
    pairs = [[query, doc] for doc in documents]
    scores = reranker_model.predict(pairs)
    
    # Sort documents by score in descending order
    ranked_docs = [doc for _, doc in sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)]
    return ranked_docs