# rag-experiments

Hands-on RAG (Retrieval-Augmented Generation) implementation series.
Built on a self-hosted VPS using LlamaIndex, Qdrant, and OpenAI API.

Each script represents one learning stage — from raw API calls to production-ready agents.

---

## Stack

| Component | Technology |
|---|---|
| RAG framework | LlamaIndex |
| Vector database | Qdrant (self-hosted) |
| LLM | OpenAI gpt-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Infrastructure | Contabo VPS, Ubuntu 22.04, Docker |

---

## Scripts

### `rag_manual.py` — P1: RAG without frameworks

Raw implementation using only OpenAI API directly.
No LlamaIndex, no abstractions — shows what happens under the hood.

- Manual chunking with overlap
- Direct embedding API calls
- Cosine similarity computed with numpy
- In-memory vector store

**What you learn:** how embeddings work, what chunking is, why overlap matters.

---

### `rag_llamaindex_qdrant.py` — P2: LlamaIndex + Qdrant

Production-grade pipeline using LlamaIndex and self-hosted Qdrant.

- SimpleDirectoryReader for PDF ingestion
- Qdrant as persistent vector store
- Collection existence check (skip re-indexing if already done)
- Interactive Q&A loop with source attribution

**What you learn:** how LlamaIndex abstracts the RAG pipeline, how Qdrant stores vectors persistently.

---

### `rag_multidoc.py` — P3: Multi-document + Metadata Filtering

Multiple documents with different types, smart incremental indexing.

- SHA256 hash registry — only new/changed files get indexed
- Explicit metadata per document (`doc_type`, `file_name`, `source`)
- Metadata filtering at query time — search only within specific document types
- `index.insert()` — adds documents without rebuilding the collection
- Source attribution with cosine similarity scores

**What you learn:** metadata filtering, incremental indexing, real-world document management.

---

## How Metadata Filtering Works

```
Query: "What are the password requirements?"
Filter: doc_type = 'policy'

Qdrant flow:
  1. Pre-filter: keep only vectors where doc_type == 'policy'
  2. Vector search within filtered subset → top-3 by cosine similarity
  3. Pass chunks to LLM as context

Result: answer comes only from policy documents,
        even if other documents are semantically closer.
```

This is pre-filtering — Qdrant restricts the search space before vector comparison,
not after. Faster and more precise than post-filtering.

---

## Cosine Similarity — Score Reference

```
score ≥ 0.6   → high relevance, document clearly contains the answer
score 0.4–0.6 → moderate relevance
score < 0.35  → low relevance, consider returning "not found" instead of hallucinating
```

---

## Setup

```bash
# Clone and enter
git clone https://github.com/leobayker/rag-experiments.git
cd rag-experiments

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install llama-index llama-index-vector-stores-qdrant \
            llama-index-embeddings-openai llama-index-llms-openai \
            qdrant-client python-dotenv openai

# Set OpenAI API key
echo "OPENAI_API_KEY=your-key-here" > .env

# Run Qdrant (Docker)
docker run -d -p 6333:6333 qdrant/qdrant

# Place documents in docs/ folder, then run
python rag_multidoc.py
```

---

## Document Type Detection

`rag_multidoc.py` determines `doc_type` from the filename:

| Keyword in filename | doc_type |
|---|---|
| `policy` | policy |
| `procedure` | procedure |
| `report`, `test` | report |
| `contract` | contract |
| anything else | general |

For production: use folder structure, a CSV manifest, or LLM-based classification.

---

## Roadmap

| Script | Stage | Status |
|---|---|---|
| `rag_manual.py` | P1 — Raw RAG | ✅ Done |
| `rag_llamaindex_qdrant.py` | P2 — LlamaIndex + Qdrant | ✅ Done |
| `rag_multidoc.py` | P3 — Multi-doc + Metadata | ✅ Done |
| `rag_multidoc_streamlit.py` | P4 — Streamlit UI | ⬜ Planned |
| `agent_react.py` | P5–P6 — ReAct Agent | ⬜ Planned |
| `agent_memory.py` | P7 — Agent with Memory | ⬜ Planned |
| FastAPI endpoint | P8 — Production API | ⬜ Planned |
| On-premise (Ollama) | P9 — No cloud dependencies | ⬜ Planned |

---

## Author

Network & Cybersecurity Engineer transitioning into RAG and AI Agents.
Building toward on-premise AI solutions for enterprise clients.

[LinkedIn](https://linkedin.com/in/artem-hrechnyi) · [Upwork](https://upwork.com/freelancers/leobayker)
