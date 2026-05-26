"""
rag.py — Retrieval layer for the portfolio RAG chatbot.

Loads the FAISS index and chunk metadata once at startup,
then exposes a single search() function used by main.py.
"""

import json
from pathlib import Path

import faiss
import numpy as np
from fastembed import TextEmbedding

# ── Paths ─────────────────────────────────────────────────────────────────────
INDEX_DIR   = Path(__file__).parent / "index"
INDEX_PATH  = INDEX_DIR / "faiss.index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"

# ── Load once at import time ───────────────────────────────────────────────────
print(f"[rag] Loading embedding model: {MODEL_NAME} ...")
_model = TextEmbedding(MODEL_NAME)

print(f"[rag] Loading FAISS index from: {INDEX_PATH}")
_index = faiss.read_index(str(INDEX_PATH))

print(f"[rag] Loading chunks from: {CHUNKS_PATH}")
with open(CHUNKS_PATH, encoding="utf-8") as f:
    _chunks: list[dict] = json.load(f)

print(f"[rag] Ready — {_index.ntotal} vectors, {len(_chunks)} chunks")


# ── Public API ────────────────────────────────────────────────────────────────
def search(query: str, top_k: int = 5) -> list[dict]:
    """
    Embed the query and return the top_k most relevant chunks.

    Each result is a dict:
        {
            "text":   str,   # the raw chunk text
            "source": str,   # e.g. "projects.md"
            "score":  float  # cosine similarity (0–1, higher is better)
        }
    """
    # Embed (fastembed normalises by default — matches how ingest.py built the index)
    vector = np.array(list(_model.embed([query]))[0], dtype="float32").reshape(1, -1)

    # Search
    scores, indices = _index.search(vector, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:           # FAISS returns -1 when fewer results exist
            continue
        chunk = _chunks[idx]
        results.append({
            "text":   chunk["text"],
            "source": chunk["source"],
            "score":  float(score),
        })

    return results
