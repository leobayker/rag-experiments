# rag-experiments

Personal learning repository — building RAG pipelines and AI Agents from scratch.
Stack: LlamaIndex · Qdrant · OpenAI API · FastAPI · Streamlit

---

## Structure

| File | Stage | Description |
|------|-------|-------------|
| `rag_manual.py` | P1 | RAG without frameworks — raw OpenAI API, cosine similarity in pure Python |
| `rag_llamaindex_qdrant.py` | P2 | RAG with LlamaIndex + Qdrant vector store |
| `rag_multidoc.py` | P3 | Multi-document RAG with metadata filtering *(coming soon)* |
| `streamlit_app.py` | P4 | Web UI for RAG pipeline *(coming soon)* |
| `agent_react.py` | P5-P6 | ReAct agent with tools *(coming soon)* |
| `agent_memory.py` | P7 | Agent with persistent memory *(coming soon)* |

---

## P1 — RAG Without Frameworks (`rag_manual.py`)

Manual implementation to understand what happens under the hood:

- Load PDF → extract text with `pypdf`
- Split into chunks manually
- Call OpenAI Embeddings API directly → get 1536-dimension vectors
- Store vectors in a Python dict (in-memory, lost on restart)
- On query: embed the question → compute cosine similarity → find nearest chunk → pass to GPT

**Purpose:** understand the mechanics before using any framework.

---

## P2 — LlamaIndex + Qdrant (`rag_llamaindex_qdrant.py`)

Same RAG pipeline, but on a production stack.

### What changed vs P1

| Problem in P1 | Solution in P2 |
|---------------|----------------|
| Vectors stored in dict — lost on restart | Qdrant vector store — persistent on disk |
| Manual chunking | LlamaIndex handles chunking automatically |
| Manual cosine similarity search | Qdrant handles similarity search |
| Single hardcoded question | Interactive terminal loop |
| 1 chunk as context for GPT | top-3 chunks (`similarity_top_k=3`) |

### Problems encountered and how they were solved

**Problem 1: All queries returned the same chunk**
- Cause: `SimpleDirectoryReader` by default merges all PDF pages into a single `Document` object → LlamaIndex creates only 1 chunk → Qdrant has 1 vector → every query returns it
- Fix: `file_extractor={".pdf": PDFReader()}` — forces page-by-page splitting. Each page = separate `Document` = separate chunks in Qdrant

**Problem 2: Re-indexing on every run (wastes API money)**
- Cause: naive approach — delete collection and re-index every time the script runs
- Fix: persistence check before indexing:
  ```python
  collections = [c.name for c in client.get_collections().collections]
  if "p2_documents" in collections:
      index = VectorStoreIndex.from_vector_store(vector_store)  # reuse
  else:
      index = VectorStoreIndex.from_documents(documents, ...)   # index once
  ```
- Result: first run indexes and stores vectors in Qdrant. All subsequent runs reuse existing vectors — instant startup, zero API cost
- Known limitation: if you add a new file to `docs/`, the check won't detect it. Delete the collection manually in Qdrant Dashboard to force re-indexing. Fixed properly in P3 with per-file metadata tracking.

**Problem 3: venv broken after folder rename**
- Cause: venv stores absolute paths internally. After renaming `rag-week1` → `rag-experiments`, `which python3` pointed to system Python despite venv being "active"
- Fix: delete and recreate venv from scratch in the new location

### How it works

```
Question (text)
    ↓
OpenAI Embeddings API → vector (1536 numbers)
    ↓
Qdrant similarity search → top-3 nearest chunks
    ↓
GPT-4o (chunks as context) → Answer
    ↓
source_nodes → show which chunks were used + similarity score
```

### Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install llama-index llama-index-vector-stores-qdrant \
            llama-index-embeddings-openai llama-index-readers-file \
            qdrant-client python-dotenv
```

Requires Qdrant running locally:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

Create `.env` file:
```
OPENAI_API_KEY=sk-...
```

Place documents in `docs/` folder, then run:
```bash
python3 rag_llamaindex_qdrant.py
```

### Score interpretation

| Score | Meaning |
|-------|---------|
| 0.7+ | Good match — chunk is relevant to the question |
| 0.5–0.7 | Acceptable — some relevance |
| below 0.4 | Poor match — wrong chunk retrieved |

Low scores usually mean: question phrasing doesn't match document content, or chunking strategy needs adjustment.

---

## Infrastructure

- **VPS:** Ubuntu 22.04, 4 vCPU / 8GB RAM
- **Qdrant:** self-hosted in Docker
- **LLM:** OpenAI GPT-4o (cloud)
- **Embeddings:** `text-embedding-3-small` (~5x cheaper than ada-002, equal quality for RAG)

---

## Roadmap

- [x] P1 — Manual RAG (no frameworks)
- [x] P2 — LlamaIndex + Qdrant
- [ ] P3 — Multi-document + metadata filtering
- [ ] P4 — Streamlit UI
- [ ] P5-P6 — ReAct Agent
- [ ] P7 — Agent memory
- [ ] P8 — FastAPI production endpoint
- [ ] P9 — On-premise RAG with Ollama (no internet required)
