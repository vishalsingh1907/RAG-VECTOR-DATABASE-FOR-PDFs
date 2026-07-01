# RAG + OKF Knowledge Base

A self-hosted RAG system that ingests PDFs into an **Open Knowledge Format** bundle (markdown + YAML frontmatter), embeds them into a vector database, and serves grounded answers through a local LLM — with a second model verifying every answer.

## Quick start

```bash
# 1. Start the stack
docker compose up -d --build

# 2. First run only — pull the models (takes a few minutes)
docker exec rag-ollama ollama pull nomic-embed-text
docker exec rag-ollama ollama pull qwen2.5:14b-instruct-q4_K_M
docker exec rag-ollama ollama pull llama3.1:8b-instruct-q4_K_M

# 3. Drop a PDF — it gets indexed automatically
cp my-datasheet.pdf ./pdfs/

# 4. Open the chat UI
#    → http://localhost:3000
```

## Architecture

| Service | Port | Role |
|---------|------|------|
| **ollama** | 11434 | LLM, embeddings, verifier (GPU) |
| **qdrant** | 6333 | Vector database |
| **ingest** | — | Watches `./pdfs/`, writes OKF, embeds |
| **api** | 8000 | FastAPI: retrieve → generate → verify |
| **webui** | 3000 | Open WebUI chat interface |

## API

### `POST /ask` — native endpoint

```json
{ "question": "What is the GPIO configuration for STM32F4?", "top_k": 6 }
```

Returns `answer`, `verification`, `grounded` (bool), and `sources`.

### `POST /v1/chat/completions` — OpenAI-compatible

Used by Open WebUI automatically. Supports streaming.

## Models (16 GB VRAM)

| Role | Model | VRAM |
|------|-------|------|
| Main LLM | `qwen2.5:14b-instruct-q4_K_M` | ~10 GB |
| Verifier | `llama3.1:8b-instruct-q4_K_M` | ~5 GB |
| Embeddings | `nomic-embed-text` | ~1.5 GB |

Ollama swaps models on demand — they don't all need to be resident simultaneously.

## OKF output

Every ingested PDF produces a human-readable, git-diffable markdown bundle:

```
okf/
├── index.md                   # lists all documents
└── my-datasheet/
    ├── index.md               # per-doc table of contents
    ├── section-001.md          # chunk with YAML frontmatter
    └── section-002.md
```

You can `cat`, `grep`, or `git diff` your entire knowledge base.

## Adding PDFs later

```bash
cp new-document.pdf ./pdfs/
```

The watcher picks it up within seconds — no restart or manual re-index needed.

## Configuration

Edit `.env` to change models, collection names, or service URLs.
