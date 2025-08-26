FROM vllm/vllm-openai:v0.8.4

WORKDIR /app
COPY app /app/app
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Only safe, minimal deps:
# - httpx for proxying to vLLM
# - sentence-transformers WITHOUT deps (use base image's transformers/torch)
RUN pip install --no-cache-dir httpx==0.27.0 && \
    pip install --no-cache-dir --no-deps sentence-transformers==3.0.1

EXPOSE 8000
ENTRYPOINT ["/app/start.sh"]
