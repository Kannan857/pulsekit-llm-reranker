FROM vllm/vllm-openai:v0.8.4

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Curl for readiness checks in start.sh
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY app /app/app
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8000

# IMPORTANT: override base entrypoint so our script runs
ENTRYPOINT ["/app/start.sh"]
