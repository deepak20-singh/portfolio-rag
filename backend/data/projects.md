# Projects

## 1. VibeCheck — Real-time Emotion Detector (Featured)
Role: Full-stack | Year: 2025
Stack: React, TypeScript, FastAPI, MediaPipe, WebSockets, Groq LLM, Recharts
Live Demo: https://vibecheck-frontend-opal.vercel.app
GitHub: https://github.com/deepak20-singh/vibecheck_frontend

Webcam-based real-time emotion detection and engagement scoring. Dual processing
modes: MediaPipe face landmarks (in-browser WASM, ~8fps, instant) and a Groq-powered
vision LLM (every 2 seconds, semantic depth). Live emotion timeline visualization.

Highlights:
- Dual-stream pipeline: MediaPipe 8fps + Groq LLM running in parallel
- Tracks 10 distinct emotions simultaneously with head pose detection (ZYX Euler)
- Privacy-first: video stays on-device, frames processed in-memory and discarded
- FastAPI backend with WebSocket sessions, IP-based rate limiting, daily quotas
- Zero persistent storage — all processing client-side or ephemeral

---

## 2. Insurance Document Extraction Platform (Flagship)
Role: Backend Engineer | Year: 2024–Present
Stack: Python, FastAPI, LLMs, Docker, AWS, Vector Search

FastAPI microservice that extracts structured JSON from insurance documents using
self-hosted LLMs, automating document understanding workflows for downstream systems.

Highlights:
- Multi-stage pipeline: conversion → classification → extraction → post-processing
- Supports PDF, Excel, and DOC inputs with modular, testable Python components
- Provider-agnostic LLM integration layer (sync/async, concurrency control)
- Vector-based search + prompt-driven extraction for accurate structured outputs
- Containerized on Docker, deployed to AWS

---

## 3. 3D AI Avatar Interaction System
Role: Lead Backend | Year: 2023–Present
Stack: FastAPI, LLMs, WebRTC, AWS, Docker

Engineered the AI backend for a real-time conversational LLM avatar. Prompt engineering,
structured output validation, and WebRTC for low-latency interactions.

Highlights:
- 10,000+ messages/hour at sub-150ms latency, horizontally scalable on AWS
- LLM output validation + RAG pipelines → 30% improvement in response reliability
- Dockerized microservices for low latency and high availability

---

## 4. Real-Time Session Management Dashboard
Role: Backend Engineer | Year: 2023
Stack: Flask, Redis, MongoDB, Microservices, CI/CD

Operator dashboard for live session telemetry. Redis-driven event bus,
RAG-grounded responses, role-based access control.

Highlights:
- Sub-100ms dashboard refresh via Redis caching + Pub/Sub
- RAG pipeline reduced hallucinations by 40%
- RBAC for access governance
- 25% drop in support tickets post-launch

---

## 5. Health Monitoring Web Application
Role: Full-stack | Year: 2023
Stack: LLMs, RAG, Flask, Redis, AWS, Azure

Emotionally responsive medical dialogue system powered by an LLM avatar grounded
in domain data. Multi-cloud deployment across AWS EC2 and Azure App Services.

Highlights:
- 10k+ message exchanges/hour via modular APIs
- 35% faster event handling after Redis tuning
- P95 < 150ms across the inference path
- Multi-cloud: AWS EC2 + Azure App Services for high availability

---

## 6. Loan RAG Assistant (GitHub)
Stack: Python, Qdrant, FastAPI, Redis, sentence-transformers, BM25, Groq (Llama-3)
GitHub: https://github.com/deepak20-singh/loan-RAG-assistant

RAG system for answering questions about small business loan policies. Uses hybrid
retrieval — vector search + BM25 + Reciprocal Rank Fusion — for more accurate
retrieval of lending-specific values (loan amounts, credit scores).

Highlights:
- Hybrid retrieval: semantic (Qdrant) + keyword (BM25) + Reciprocal Rank Fusion
- sentence-transformers embeddings (BAAI/bge-small-en-v1.5)
- Planned: Groq LLM integration, Pydantic structured outputs, Redis caching
