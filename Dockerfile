FROM vllm/vllm-openai:v0.8.4

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 1) Dep layer stays cacheable
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
# requirements.txt should NOT include: torch, vllm

# 2) App code
COPY app /app/app
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8000
ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["/bin/bash","/app/start.sh"]
