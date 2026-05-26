"""
ingest.py — Build the FAISS knowledge base for the portfolio RAG chatbot.

Usage:
    python ingest.py

Output:
    index/faiss.index  — FAISS inner-product index (cosine sim after L2-norm)
    index/chunks.json  — chunk texts + source filenames
"""

import json
import os
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent / "data"
INDEX_DIR  = Path(__file__).parent / "index"
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE    = 500   # characters — safe for 256-token MiniLM limit
CHUNK_OVERLAP = 80    # characters — context bleed between adjacent chunks
MIN_CHUNK_LEN = 30    # discard tiny fragments (headers, blank lines, etc.)


# ── Chunker ───────────────────────────────────────────────────────────────────
def _split_recursive(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """Recursively split text preferring coarser boundaries first."""
    if not text.strip():
        return []
    if len(text) <= chunk_size or not separators:
        return [text.strip()]

    sep, rest = separators[0], separators[1:]
    parts = text.split(sep)

    chunks: list[str] = []
    current = ""

    for part in parts:
        candidate = (current + sep + part).strip() if current else part.strip()
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # Part itself may be too big — recurse
            if len(part.strip()) > chunk_size:
                chunks.extend(_split_recursive(part.strip(), rest, chunk_size))
                current = ""
            else:
                current = part.strip()

    if current:
        chunks.append(current)

    return chunks


def split_text(text: str) -> list[str]:
    """Split a document into overlapping chunks."""
    raw = _split_recursive(
        text,
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=CHUNK_SIZE,
    )

    # Apply overlap: prepend the tail of the previous chunk
    result: list[str] = []
    for i, chunk in enumerate(raw):
        if i > 0 and CHUNK_OVERLAP:
            tail = raw[i - 1][-CHUNK_OVERLAP:]
            chunk = (tail + " " + chunk).strip()
        result.append(chunk)

    return result


# ── Document loader ───────────────────────────────────────────────────────────
def load_documents(data_dir: Path) -> list[dict]:
    """Read all .md files; return list of {text, source}."""
    docs = []
    for md in sorted(data_dir.glob("*.md")):
        text = md.read_text(encoding="utf-8").strip()
        if text:
            docs.append({"text": text, "source": md.name})
            print(f"  loaded  {md.name}  ({len(text):,} chars)")
    return docs


def chunk_documents(docs: list[dict]) -> list[dict]:
    """Split each document into chunks, carrying the source filename."""
    chunks = []
    for doc in docs:
        for part in split_text(doc["text"]):
            if len(part) >= MIN_CHUNK_LEN:
                chunks.append({"text": part, "source": doc["source"]})
    return chunks


# ── Embedder & indexer ────────────────────────────────────────────────────────
def embed_and_save(chunks: list[dict]) -> None:
    """Embed with sentence-transformers → FAISS IndexFlatIP → save to disk."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading model  : {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    texts = [c["text"] for c in chunks]
    print(f"Embedding      : {len(texts)} chunks ...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2-norm → inner product == cosine sim
        batch_size=64,
    )
    embeddings = np.array(embeddings, dtype="float32")

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # cosine similarity (vectors are normalised)
    index.add(embeddings)

    index_path  = INDEX_DIR / "faiss.index"
    chunks_path = INDEX_DIR / "chunks.json"

    faiss.write_index(index, str(index_path))
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Index saved  → {index_path}  ({index.ntotal} vectors, dim={dim})")
    print(f"✓ Chunks saved → {chunks_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Portfolio RAG — Ingest Pipeline")
    print("=" * 55)

    print(f"\n[1/3] Loading documents from: {DATA_DIR}")
    docs = load_documents(DATA_DIR)
    print(f"      → {len(docs)} files loaded")

    print(f"\n[2/3] Chunking  (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    chunks = chunk_documents(docs)
    print(f"      → {len(chunks)} chunks produced")

    # Quick size report
    lens = [len(c["text"]) for c in chunks]
    print(f"      → avg {sum(lens)//len(lens)} chars | "
          f"min {min(lens)} | max {max(lens)}")

    print("\n[3/3] Embedding + indexing")
    embed_and_save(chunks)

    print("\n✅  Done — knowledge base is ready.")
