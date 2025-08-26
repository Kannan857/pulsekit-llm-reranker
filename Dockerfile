FROM vllm/vllm-openai:v0.8.4

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install tini so ENTRYPOINT works
RUN apt-get update && apt-get install -y --no-install-recommends tini ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY app /app/app
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8000

# tini will properly forward signals and reap zombies
ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["/app/start.sh"]
