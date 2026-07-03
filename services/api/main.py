"""
main.py — RAG + OKF query API with hallucination verification

Endpoints:
  POST /ask                     — retrieve → generate → verify (native)
  POST /v1/chat/completions     — OpenAI-compatible wrapper (for Open WebUI)
  GET  /v1/models               — model list (for Open WebUI discovery)
  GET  /health                  — liveness probe
"""

import os
import time
import json
import logging
import uuid
from pathlib import Path
from textwrap import dedent

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

# ── Configuration ────────────────────────────────────────────────────
OLLAMA     = os.environ["OLLAMA_HOST"]
COLLECTION = os.environ["COLLECTION_NAME"]
LLM_MODEL      = os.environ["LLM_MODEL"]
VERIFIER_MODEL = os.environ["VERIFIER_MODEL"]
EMBED_MODEL    = os.environ["EMBED_MODEL"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [api] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("api")

# ── Retry-aware HTTP session ─────────────────────────────────────────
_session = requests.Session()
_session.mount(
    "http://",
    HTTPAdapter(max_retries=Retry(total=4, backoff_factor=1,
                                  status_forcelist=[502, 503, 504])),
)

# ── Qdrant client ───────────────────────────────────────────────────
qdrant = QdrantClient(url=os.environ["QDRANT_HOST"])

# ── FastAPI ──────────────────────────────────────────────────────────
app = FastAPI(title="RAG-OKF API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ─────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    top_k: int = Field(default=6, ge=1, le=30)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage] = []
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = False


# ── Core helpers ─────────────────────────────────────────────────────

def pull_model_if_missing(model: str) -> None:
    """Pull a model via Ollama if not already downloaded."""
    try:
        tags = _session.get(f"{OLLAMA}/api/tags", timeout=10).json()
        names = [m["name"] for m in tags.get("models", [])]
        if any(model in n for n in names):
            return
    except Exception:
        pass
    log.info("Pulling model '%s' …", model)
    _session.post(f"{OLLAMA}/api/pull",
                  json={"name": model, "stream": False}, timeout=600)
    log.info("Model '%s' ready.", model)


def embed(text: str) -> list[float]:
    r = _session.post(
        f"{OLLAMA}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["embedding"]


def chat_ollama(model: str, prompt: str, temperature: float = 0.4) -> str:
    """Synchronous single-turn generation."""
    r = _session.post(
        f"{OLLAMA}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["response"]


def chat_ollama_stream(model: str, prompt: str, temperature: float = 0.4):
    """Streaming single-turn generation — yields text chunks."""
    r = _session.post(
        f"{OLLAMA}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": temperature},
        },
        stream=True,
        timeout=300,
    )
    r.raise_for_status()
    for line in r.iter_lines():
        if line:
            data = json.loads(line)
            chunk = data.get("response", "")
            if chunk:
                yield chunk
            if data.get("done"):
                break


def retrieve(question: str, top_k: int = 6) -> list[dict]:
    """Embed a question and retrieve top-k chunks from Qdrant."""
    qvec = embed(question)
    hits = qdrant.query_points(
        collection_name=COLLECTION,
        query=qvec,
        limit=top_k,
    ).points
    return [
        {
            "title": h.payload.get("title", ""),
            "resource": h.payload.get("resource", ""),
            "text": h.payload.get("text", ""),
            "score": h.score,
        }
        for h in hits
    ]


def build_context_block(sources: list[dict]) -> str:
    """Format retrieved chunks into a single context string for the LLM."""
    parts = []
    for s in sources:
        parts.append(f"[{s['title']}] ({s['resource']})\n{s['text']}")
    return "\n\n---\n\n".join(parts)


def build_generation_prompt(context: str, question: str) -> str:
    return dedent(f"""\
        Answer the question using ONLY the context below.
        Cite the source in brackets after each claim, e.g. [Document Title, pages 12-14].
        If the context does not contain the answer, say "I am designed to only answer questions based on the provided database, and your query was not found in it."

        Context:
        {context}

        Question: {question}

        Answer:""")


def build_verify_prompt(context: str, draft: str) -> str:
    return dedent(f"""\
        You are a strict fact-checker.  Given the context and a draft answer,
        check whether EVERY factual claim in the answer is directly supported
        by the context.

        Respond with exactly one of:
        • "GROUNDED" — if every claim is supported.
        • "UNGROUNDED: <list the unsupported claims>" — otherwise.

        Context:
        {context}

        Draft answer:
        {draft}

        Verdict:""")


# ── Startup: ensure models exist ─────────────────────────────────────

@app.on_event("startup")
def _startup():
    for model in (EMBED_MODEL, LLM_MODEL, VERIFIER_MODEL):
        pull_model_if_missing(model)
    log.info("All models ready.")


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest):
    """
    Full RAG pipeline: retrieve → generate → verify.
    Returns the answer, verification verdict, and sources.
    """
    sources = retrieve(req.question, req.top_k)
    if not sources:
        return {
            "answer": "I am designed to only answer questions based on the provided database, and your query was not found in it.",
            "verification": "N/A",
            "grounded": False,
            "sources": [],
        }

    context = build_context_block(sources)

    # ── Generate ──
    gen_prompt = build_generation_prompt(context, req.question)
    draft = chat_ollama(LLM_MODEL, gen_prompt)

    # ── Verify ──
    verify_prompt = build_verify_prompt(context, draft)
    verification = chat_ollama(VERIFIER_MODEL, verify_prompt, temperature=0.1)
    grounded = verification.strip().upper().startswith("GROUNDED")

    return {
        "answer": draft,
        "verification": verification,
        "grounded": grounded,
        "sources": [
            {"title": s["title"], "resource": s["resource"], "score": s["score"]}
            for s in sources
        ],
    }


# ── OpenAI-compatible endpoints (for Open WebUI) ────────────────────

@app.get("/v1/models")
def list_models():
    """Return a model list so Open WebUI discovers our RAG pipeline."""
    return {
        "object": "list",
        "data": [
            {
                "id": "rag-okf",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions.
    Extracts the last user message, runs the full RAG pipeline,
    and returns the answer in the expected format.
    """
    # Extract user question from the messages
    question = ""
    for msg in reversed(req.messages):
        if msg.role == "user":
            question = msg.content
            break

    if not question:
        question = "Hello"

    # Retrieve context
    sources = retrieve(question, top_k=6)
    context = build_context_block(sources) if sources else ""

    gen_prompt = build_generation_prompt(context, question)

    # ── Non-streaming ──
    if not req.stream:
        draft = chat_ollama(LLM_MODEL, gen_prompt, temperature=req.temperature)

        # Run verification in the background (include result in metadata)
        verification = ""
        grounded = True
        if context:
            verify_prompt = build_verify_prompt(context, draft)
            verification = chat_ollama(VERIFIER_MODEL, verify_prompt, temperature=0.1)
            grounded = verification.strip().upper().startswith("GROUNDED")

        # Append citation block
        citation_block = "\n\n---\n**Sources:**\n" + "\n".join(
            f"- {s['title']} (`{s['resource']}`) — score {s['score']:.3f}"
            for s in sources
        ) if sources else ""

        if not grounded:
            citation_block += f"\n\n⚠️ **Verification warning:** {verification}"

        answer = draft + citation_block

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "rag-okf",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    # ── Streaming ──
    def _stream():
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        for chunk_text in chat_ollama_stream(LLM_MODEL, gen_prompt, temperature=req.temperature):
            payload = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "rag-okf",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk_text},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(payload)}\n\n"

        # Append sources at the end
        if sources:
            source_text = "\n\n---\n**Sources:**\n" + "\n".join(
                f"- {s['title']} (`{s['resource']}`) — score {s['score']:.3f}"
                for s in sources
            )
            payload = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "rag-okf",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": source_text},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(payload)}\n\n"

        # Final chunk
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": "rag-okf",
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": "stop"}
            ],
        }
        yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")
