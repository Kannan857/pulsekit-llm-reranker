# 🤖 pulsekit-inference-service

This repository contains the backend inference service for **pulsekit**. It hosts and serves a large language model (Llama 3.1) and a reranker model through a high-performance, low-latency REST API.

---

## ✨ Features

-   **Optimized LLM Serving:** Serves Llama 3.1 using vLLM for high-throughput and low-latency inference.
-   **Reranker Endpoint:** Provides an endpoint to intelligently rerank documents for search and retrieval tasks.
-   **Standalone Microservice:** Designed to be called by other applications, such as a voice webhook handler or a chat frontend.

---

## 🛠️ Tech Stack

-   **Backend:** FastAPI (Python)
-   **LLM Serving:** vLLM
-   **Models:**
    -   LLM: `meta-llama/Meta-Llama-3.1-8B-Instruct-AWQ`
    -   Reranker: `BAAI/bge-reranker-base`
-   **Deployment:** Docker on RunPod

---

## 🚀 Getting Started

### Prerequisites

-   Python 3.10+
-   NVIDIA GPU with CUDA 12.1+
-   Docker

### Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/pulsekit-inference-service.git](https://github.com/your-username/pulsekit-inference-service.git)
    cd pulsekit-inference-service
    ```

2.  **Create and fill environment variables:**
    ```bash
    cp .env.example .env
    ```
    Now, edit the `.env` file with your API keys if needed.

3.  **Build and run the Docker container:**
    ```bash
    docker build -t pulsekit-inference .
    docker run -p 8000:8000 --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface -env-file .env pulsekit-inference
    ```
    *Note: The `-v` flag mounts your Hugging Face cache to avoid re-downloading models.*

---

## 🔌 API Endpoints

-   `GET /health`: Health check.
-   `POST /chat`: Send a prompt and get a chat response from the LLM.
-   `POST /rerank`: Rerank a list of documents for a given query.