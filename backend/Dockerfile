# ── Base ──────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# libgomp1 is required by faiss-cpu and ONNX Runtime (fastembed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Install dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-download the embedding model ──────────────────────────────────────────
# Bakes all-MiniLM-L6-v2 into the image so runtime never needs a network call.
RUN python -c "\
from fastembed import TextEmbedding; \
list(TextEmbedding('sentence-transformers/all-MiniLM-L6-v2').embed(['warmup'])); \
print('Model cached successfully')"

# ── Copy application code ──────────────────────────────────────────────────────
COPY . .

EXPOSE 8000

# PORT is set by Render at runtime; falls back to 8000 locally
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
