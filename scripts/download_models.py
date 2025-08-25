# scripts/download_models.py
import os
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

load_dotenv()

def download():
    """Downloads the LLM and reranker models from Hugging Face."""
    llm_model_name = os.getenv("LLM_MODEL")
    reranker_model_name = os.getenv("RERANKER_MODEL")

    print(f"Downloading LLM: {llm_model_name}...")
    snapshot_download(repo_id=llm_model_name, allow_patterns=["*.json", "*.safetensors", "*.model"])
    
    print(f"Downloading Reranker: {reranker_model_name}...")
    snapshot_download(repo_id=reranker_model_name)
    
    print("All models downloaded successfully.")

if __name__ == "__main__":
    download()