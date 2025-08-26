# Lean on the official vLLM GPU image (CUDA 12, Python preinstalled)
# https://hub.docker.com/r/vllm/vllm-openai  (pick a pinned tag, e.g. v0.8.4)
FROM vllm/vllm-openai:v0.8.4

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps kept minimal
RUN apt-get update && apt-get install -y --no-install-recommends \
    tini ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY app /app/app
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8000
ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["/bin/bash","/app/start.sh"]
