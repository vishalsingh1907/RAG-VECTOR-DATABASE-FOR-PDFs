"""
watcher.py — PDF watcher → OKF bundle → embed → Qdrant

Watches /data/pdfs for new .pdf files.  For each one it:
  1. Extracts text with PyMuPDF (page-grouped chunks).
  2. Writes an OKF markdown bundle under /data/okf/<doc-id>/.
  3. Embeds each chunk via Ollama and upserts into Qdrant.

On first boot it also processes any PDFs already present in the folder.
"""

import os
import re
import time
import hashlib
import logging
from pathlib import Path
from textwrap import dedent

import fitz  # PyMuPDF
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from watchdog.observers.polling import PollingObserver as Observer
from watchdog.events import FileSystemEventHandler
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# ── Configuration ────────────────────────────────────────────────────
PDF_DIR   = Path("/data/pdfs")
OKF_DIR   = Path("/data/okf")
OLLAMA    = os.environ["OLLAMA_HOST"]
EMBED_MODEL = os.environ["EMBED_MODEL"]
COLLECTION  = os.environ["COLLECTION_NAME"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ingest] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest")

# ── Retry-aware HTTP session (Ollama may still be loading a model) ──
_session = requests.Session()
_session.mount(
    "http://",
    HTTPAdapter(max_retries=Retry(total=6, backoff_factor=2,
                                  status_forcelist=[502, 503, 504])),
)

# ── Qdrant client ───────────────────────────────────────────────────
qdrant = QdrantClient(url=os.environ["QDRANT_HOST"])


# ── Helpers ──────────────────────────────────────────────────────────

def pull_model_if_missing(model: str) -> None:
    """Ask Ollama to pull a model if it isn't already present."""
    try:
        tags = _session.get(f"{OLLAMA}/api/tags", timeout=10).json()
        names = [m["name"] for m in tags.get("models", [])]
        if any(model in n for n in names):
            log.info("Model '%s' already present.", model)
            return
    except Exception:
        pass  # if we can't check, just try pulling

    log.info("Pulling model '%s' (first run only) …", model)
    r = _session.post(
        f"{OLLAMA}/api/pull",
        json={"name": model, "stream": False},
        timeout=600,
    )
    r.raise_for_status()
    log.info("Model '%s' ready.", model)


def ensure_collection(dim: int = 768) -> None:
    """Create the Qdrant collection if it doesn't exist yet."""
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        log.info("Created Qdrant collection '%s' (dim=%d).", COLLECTION, dim)
    else:
        log.info("Qdrant collection '%s' already exists.", COLLECTION)


def embed(text: str) -> list[float]:
    """Get an embedding vector from Ollama."""
    r = _session.post(
        f"{OLLAMA}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["embedding"]


def _stable_id(doc_id: str, fname: str) -> str:
    """Deterministic hex-string point ID for a (doc, section) pair."""
    return hashlib.md5(f"{doc_id}/{fname}".encode()).hexdigest()


def _detect_heading(text: str) -> str | None:
    """Try to pull a heading from the first few lines of a text block."""
    for line in text.strip().splitlines()[:5]:
        line = line.strip()
        if 8 < len(line) < 120 and not line.startswith("•"):
            return line
    return None


# ── PDF → chunks ────────────────────────────────────────────────────

def chunk_pdf(pdf_path: Path, pages_per_chunk: int = 3):
    """
    Yield (chunk_text, start_page_0idx, label) for groups of pages.

    This is a straightforward page-grouped splitter.  For production use
    swap in a heading-aware or table-aware splitter (e.g. unstructured.io,
    camelot for tables).
    """
    doc = fitz.open(str(pdf_path))
    total = len(doc)
    buf: list[str] = []
    start_page = 0

    for i, page in enumerate(doc):
        text = page.get_text("text")
        buf.append(text)

        if len(buf) >= pages_per_chunk or i == total - 1:
            chunk_text = "\n".join(buf)
            heading = _detect_heading(chunk_text)
            label = heading or f"pages {start_page + 1}–{i + 1}"
            yield (chunk_text, start_page, label)
            buf = []
            start_page = i + 1

    doc.close()


# ── OKF bundle writer ───────────────────────────────────────────────

def write_okf_and_index(doc_id: str, pdf_path: Path) -> None:
    """Parse a PDF, write OKF markdown, embed chunks, upsert to Qdrant."""
    doc_dir = OKF_DIR / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    sections: list[tuple[str, str]] = []       # (filename, label)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    for n, (text, page, label) in enumerate(chunk_pdf(pdf_path), start=1):
        fname = f"section-{n:03d}.md"
        safe_title = label.replace('"', '\\"')

        content = dedent(f"""\
            ---
            type: document-section
            title: "{safe_title}"
            description: "Chunk {n} of {doc_id}"
            resource: /data/pdfs/{pdf_path.name}#page={page + 1}
            tags: []
            timestamp: {now}
            ---

            # {label}

            {text}
        """)
        (doc_dir / fname).write_text(content, encoding="utf-8")
        sections.append((fname, label))

        # ── Embed & upsert ──
        try:
            vec = embed(text[:8000])  # respect embedding-model context limit
            qdrant.upsert(
                collection_name=COLLECTION,
                points=[
                    PointStruct(
                        id=_stable_id(doc_id, fname),
                        vector=vec,
                        payload={
                            "doc_id": doc_id,
                            "file": str(doc_dir / fname),
                            "title": label,
                            "resource": f"/data/pdfs/{pdf_path.name}#page={page + 1}",
                            "text": text[:4000],  # store first 4k chars for retrieval context
                        },
                    )
                ],
            )
        except Exception:
            log.exception("Failed to embed/upsert chunk %s/%s", doc_id, fname)

    # ── Per-document index.md ──
    section_links = "\n".join(f"- [{lbl}](./{fn})" for fn, lbl in sections)
    doc_index = dedent(f"""\
        ---
        type: document-index
        title: "{doc_id}"
        resource: /data/pdfs/{pdf_path.name}
        tags: []
        timestamp: {now}
        ---

        # {doc_id}

        Source: `{pdf_path.name}` — {len(sections)} section(s), ingested {now}

        ## Sections
        {section_links}
    """)
    (doc_dir / "index.md").write_text(doc_index, encoding="utf-8")

    rebuild_bundle_index()
    log.info("Finished: %s → %d chunks.", doc_id, len(sections))


def rebuild_bundle_index() -> None:
    """Re-generate the top-level okf/index.md that lists all documents."""
    docs = sorted(p.name for p in OKF_DIR.iterdir() if p.is_dir())
    links = "\n".join(f"- [{d}](./{d}/index.md)" for d in docs)
    content = dedent(f"""\
        ---
        type: bundle-index
        okf_version: "0.1"
        title: Knowledge base
        ---

        # Knowledge base

        {links}
    """)
    (OKF_DIR / "index.md").write_text(content, encoding="utf-8")


# ── File-system watcher ─────────────────────────────────────────────

class PDFHandler(FileSystemEventHandler):
    """React to new .pdf files appearing in the watched directory."""

    def on_created(self, event):
        if event.is_directory or not event.src_path.lower().endswith(".pdf"):
            return
        time.sleep(2)  # let the file finish writing / copying
        pdf_path = Path(event.src_path)
        doc_id = pdf_path.stem
        log.info("New PDF detected: %s → doc_id=%s", pdf_path.name, doc_id)
        try:
            write_okf_and_index(doc_id, pdf_path)
        except Exception:
            log.exception("Error processing %s", pdf_path.name)


# ── Entrypoint ───────────────────────────────────────────────────────

def main() -> None:
    log.info("Starting ingest service …")

    # Pull embedding model on first boot
    pull_model_if_missing(EMBED_MODEL)

    # Ensure Qdrant collection exists
    ensure_collection()

    # Process PDFs already present in the folder
    existing = sorted(PDF_DIR.glob("*.pdf"))
    if existing:
        log.info("Found %d existing PDF(s) — indexing …", len(existing))
        for pdf in existing:
            try:
                write_okf_and_index(pdf.stem, pdf)
            except Exception:
                log.exception("Error processing existing PDF %s", pdf.name)

    # Start watching for new PDFs
    observer = Observer()
    observer.schedule(PDFHandler(), str(PDF_DIR), recursive=False)
    observer.start()
    log.info("Watching %s for new PDFs …", PDF_DIR)

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
