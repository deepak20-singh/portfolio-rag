"""
main.py — FastAPI chat API for the portfolio RAG chatbot.

Endpoints:
    GET  /health      — liveness probe (used by Render)
    POST /chat        — RAG-powered chat with SSE streaming
"""

import json
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from groq import Groq
from pydantic import BaseModel

import rag

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
TOP_K        = int(os.getenv("TOP_K", "5"))

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")

groq_client = Groq(api_key=GROQ_API_KEY)

# ── CORS origins ──────────────────────────────────────────────────────────────
CORS_ORIGINS = [
    "https://myportfolio-dun-psi.vercel.app",
    "http://localhost:5173",
    "http://localhost:3000",
]

CORS_HEADERS = {
    "Access-Control-Allow-Origin":  "https://myportfolio-dun-psi.vercel.app",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Max-Age":       "86400",
}

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Portfolio RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str

# ── Prompt builder ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful assistant embedded in a software engineer's portfolio website.
Your job is to answer questions about the engineer using ONLY the context provided below.

### STRICT GROUND RULES:
1. USE ONLY THE PROVIDED CONTEXT. Do not use any outside knowledge, assumptions, or extrapolations.
2. ADOPT THE PERSONA: Speak in the first person ("I", "my", "me") as if you are the engineer themselves.
3. TONE: Be concise, friendly, grounded, and professional.
4. ABSOLUTE TRUTH: If the answer cannot be directly and fully verified by the provided context, you MUST reply exactly with:
   "I don't have that information — feel free to reach out via the contact form."
5. NO HALLUCINATIONS: Never invent dates, technologies, projects, or experiences. If it isn't explicitly written below, it does not exist.

### EVALUATION STEP (Internal Monologue):
Before generating your final response, silently verify if every single fact you are about to state is explicitly written in the context. If even a small detail is missing, fall back to the default refusal message.

### FORMATTING:
- Use **bold** for key technologies, metrics, and important terms
- Use bullet points for any list of 3 or more items
- Keep responses concise — 2 to 5 sentences or a short bullet list
- Never use headers (##) — this is a chat widget, not a document
- Never use code blocks unless specifically asked about code"""

def build_prompt(question: str, chunks: list[dict]) -> list[dict]:
    context = "\n\n---\n\n".join(
        f"[{c['source']}]\n{c['text']}" for c in chunks
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}",
        },
    ]

# ── SSE generator ─────────────────────────────────────────────────────────────
def sse_event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"

def stream_response(question: str):
    # 1. Retrieve relevant chunks
    chunks = rag.search(question, top_k=TOP_K)
    if not chunks:
        yield sse_event({"token": "I couldn't find relevant information. Try rephrasing your question."})
        yield sse_event({"done": True})
        return

    # 2. Build prompt
    messages = build_prompt(question, chunks)

    # 3. Stream from Groq
    stream = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.3,       # low temp — factual, consistent answers
        max_tokens=512,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield sse_event({"token": delta})

    # 4. Signal completion + send sources
    sources = list({c["source"].replace(".md", "") for c in chunks})
    yield sse_event({"done": True, "sources": sources})

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": GROQ_MODEL, "vectors": rag._index.ntotal}

@app.post("/chat")
def chat(req: ChatRequest):
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if len(q) > 500:
        raise HTTPException(status_code=400, detail="Question too long (max 500 chars).")

    return StreamingResponse(
        stream_response(q),
        media_type="text/event-stream",
        headers={
            "Cache-Control":                "no-cache",
            "X-Accel-Buffering":            "no",
            "Access-Control-Allow-Origin":  "https://myportfolio-dun-psi.vercel.app",
            "Access-Control-Allow-Headers": "Content-Type",
        },
    )
