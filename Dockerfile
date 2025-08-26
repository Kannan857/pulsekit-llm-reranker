FROM vllm/vllm-openai:v0.8.4

WORKDIR /app
COPY app /app/app
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Only httpx for proxying to vLLM; transformers/torch come from base image.
RUN pip install --no-cache-dir httpx==0.27.0

EXPOSE 8000
ENTRYPOINT ["/app/start.sh"]
