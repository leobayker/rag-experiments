# rag-experiments

Practical RAG (Retrieval-Augmented Generation) and AI Agents learning path.
Built on LlamaIndex + Qdrant + OpenAI, self-hosted on VPS.

## Stack

- **LlamaIndex** — RAG pipeline
- **Qdrant** — vector database (self-hosted)
- **OpenAI API** — embeddings (`text-embedding-3-small`) + LLM (`gpt-4o-mini`)
- **Streamlit** — web UI
- **FastAPI** — production API (P8)
- **Ollama** — local models for on-premise (P9)

---

## Learning Path

| # | Project | Status | Key concepts |
|---|---|---|---|
| P1 | Manual RAG pipeline | ✅ Done | embeddings, cosine similarity, chunking |
| P2 | LlamaIndex + Qdrant | ✅ Done | vector store, persistence, query engine |
| P3 | Multi-doc + metadata filtering | ✅ Done | metadata, pre-filtering, hash registry |
| P4 | Streamlit UI + 2FA | ✅ Done | web UI, TOTP auth, nginx subpath proxy |
| P5–P6 | ReAct Agent | ⬜ | tools, reasoning loop, action-observation |
| P7 | Agent memory | ⬜ | short/long-term memory, context |
| P8 | FastAPI production | ⬜ | REST API, async, Swagger |
| P9 | On-premise RAG | ⬜ | Ollama, bge-m3, no internet required |

---

## P1 — Manual RAG (`rag_manual.py`)

RAG pipeline built from scratch without frameworks.

**What it does:**
- Loads a text document and splits it into chunks
- Generates embeddings via OpenAI API
- Stores vectors in memory (dict)
- On query: embeds the question, finds top-k similar chunks via cosine similarity
- Passes context + question to GPT for answer generation

**Key concepts learned:**
- What embeddings are (1536-dimensional vectors representing semantic meaning)
- Cosine similarity: 0.6+ relevant, below 0.35 unrelated
- Why chunking matters — chunk size directly affects answer quality
- The full RAG loop: index → retrieve → augment → generate

---

## P2 — LlamaIndex + Qdrant (`rag_llamaindex_qdrant.py`)

Same pipeline, now using production-grade tools.

**What it does:**
- Loads PDF via `SimpleDirectoryReader` with explicit `PDFReader`
- Chunks with `SentenceSplitter` (chunk_size=512, overlap=50)
- Stores vectors in Qdrant (self-hosted, persisted to disk)
- Checks if collection exists before indexing — saves API costs on re-runs
- Queries via `VectorStoreIndex.as_query_engine()`

**Key concepts learned:**
- Why `file_extractor={".pdf": PDFReader()}` is required (SimpleDirectoryReader merges PDF pages by default)
- Qdrant persistence — vectors survive process restart
- Collection existence check pattern

---

## P3 — Multi-doc + Metadata Filtering (`rag_multidoc.py`)

Production-ready multi-document RAG with metadata pre-filtering.

**What it does:**
- Indexes multiple documents from `docs/` directory
- Assigns metadata per document: `doc_type`, `source`, `filename`
- Pre-filtering: narrows Qdrant search space BEFORE vector comparison
- Hash registry (`doc_registry.json`): tracks SHA256 of each file, re-indexes only new/changed files
- Supports incremental indexing with `index.insert()`

**Key concepts learned:**
- Metadata stored alongside vectors in Qdrant payload
- Pre-filtering vs post-filtering (pre is faster and more accurate)
- Incremental indexing without full collection rebuild
- Hash-based change detection for large document sets
- Metadata key in Qdrant: `file_name` (not `filename`) — always verify with payload inspection

---

## P4 — Streamlit UI + 2FA (`streamlit_app.py`)

Web interface for the RAG pipeline with two-factor authentication.

**What it does:**
- Login form: password + TOTP code (Google Authenticator compatible)
- After auth: full RAG UI with document type filter, question input, answer display
- Sources panel: shows filename, doc_type, similarity score, text preview for each retrieved chunk
- Sidebar: collection info, embedding model, LLM info, logout button
- Session-based auth: re-login required when browser tab is closed

**Authentication flow:**
```
User enters password + 6-digit TOTP code
        ↓
server verifies password (env var RAG_PASSWORD)
        ↓
pyotp.TOTP.verify() checks code with valid_window=1 (±30 sec drift tolerance)
        ↓
st.session_state["authenticated"] = True
        ↓
Main RAG UI shown
```

**Deployment:**
- Runs as systemd service (`streamlit-rag.service`)
- Accessible via `https://leobayker.duckdns.org/rag/`
- nginx reverse proxy with `^~` location prefix match
- `--server.baseUrlPath /rag` + `proxy_pass` without trailing slash = correct path handling

**Key concepts learned:**
- Streamlit re-runs entire script on every interaction → `@st.cache_resource` is critical
- WebSocket (`/_stcore/stream`) requires separate nginx location block
- `proxy_pass host:port` (no slash) vs `proxy_pass host:port/` (with slash) — different path handling
- TOTP: `valid_window=1` compensates for clock drift between server and phone
- `st.form()` prevents script re-run on every keystroke

**Tech:**
- `pyotp` — TOTP generation and verification
- `qrcode[pil]` — QR code generation for authenticator app setup

---

## Setup

```bash
# Clone and create venv
git clone https://github.com/leobayker/rag-experiments.git
cd rag-experiments
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install llama-index llama-index-vector-stores-qdrant \
            llama-index-embeddings-openai qdrant-client \
            python-dotenv streamlit pyotp qrcode[pil]

# Configure
cp .env.example .env
# Edit .env: add OPENAI_API_KEY, RAG_PASSWORD, RAG_TOTP_SECRET

# Run Qdrant (Docker)
docker run -d -p 6333:6333 qdrant/qdrant

# Index documents
python rag_multidoc.py

# Run UI
streamlit run streamlit_app.py
```

---

## Environment Variables (`.env`)

```
OPENAI_API_KEY=sk-...
RAG_PASSWORD=your_password
RAG_TOTP_SECRET=your_base32_totp_secret
```

Generate TOTP secret:
```bash
python3 -c "import pyotp; print(pyotp.random_base32())"
```

---

## .gitignore

```
.env
*.pdf
qdrant_storage/
doc_registry.json
totp_qr.png
__pycache__/
venv/
```
