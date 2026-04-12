# rag-experiments

Practical RAG (Retrieval-Augmented Generation) and AI Agents learning path.
Built on LlamaIndex + Qdrant + OpenAI, self-hosted on VPS.

## Stack

* **LlamaIndex** — RAG pipeline + AI agents
* **Qdrant** — vector database (self-hosted)
* **OpenAI API** — embeddings (`text-embedding-3-small`) + LLM (`gpt-4o-mini`)
* **RAGAS** — RAG evaluation framework
* **Streamlit** — web UI
* **FastAPI** — production API (P7)
* **Ollama** — local models for on-premise (P8)

---

## Learning Path

| # | Project | Status | Key concepts |
| --- | --- | --- | --- |
| P1 | Manual RAG pipeline | ✅ Done | embeddings, cosine similarity, chunking |
| P2 | LlamaIndex + Qdrant | ✅ Done | vector store, persistence, query engine |
| P3 | Multi-doc + metadata filtering | ✅ Done | metadata, pre-filtering, hash registry |
| P4 | Streamlit UI + 2FA | ✅ Done | web UI, TOTP auth, nginx subpath proxy |
| P4.1 | Ukrainian language tuning | ✅ Done | HyDE, morphological snippet extraction |
| P4.2 | RAG Evaluation — RAGAS | ✅ Done | faithfulness, answer relevancy, context precision/recall |
| P5 | AI Agent with tools | ✅ Done | FunctionAgent, tool routing, async, sequential reasoning |
| P5.5 | MCP: standardized tool integration | ⬜ | MCP server/client, STDIO vs HTTP |
| P6 | Agent memory | ⬜ | buffer/summary/vector memory, chat history |
| P6.5 | Multi-model pipelines | ⬜ | cost routing, fallback chains |
| P7 | FastAPI production | ⬜ | REST API, session-based agents, multi-user |
| P7.5 | PR Review Agent + LLM Security | ⬜ | prompt injection, red-teaming, OWASP LLM Top 10 |
| P8 | On-premise RAG | ⬜ | Ollama, bge-m3, no internet required |

---

## P1 — Manual RAG (`rag_manual.py`)

RAG pipeline built from scratch without frameworks.

**What it does:**

* Loads a text document and splits it into chunks
* Generates embeddings via OpenAI API
* Stores vectors in memory (dict)
* On query: embeds the question, finds top-k similar chunks via cosine similarity
* Passes context + question to GPT for answer generation

**Key concepts learned:**

* What embeddings are (1536-dimensional vectors representing semantic meaning)
* Cosine similarity: 0.6+ relevant, below 0.35 unrelated
* Why chunking matters — chunk size directly affects answer quality
* The full RAG loop: index → retrieve → augment → generate

---

## P2 — LlamaIndex + Qdrant (`rag_llamaindex_qdrant.py`)

Same pipeline, now using production-grade tools.

**What it does:**

* Loads PDF via `SimpleDirectoryReader` with explicit `PDFReader`
* Chunks with `SentenceSplitter` (chunk\_size=512, overlap=50)
* Stores vectors in Qdrant (self-hosted, persisted to disk)
* Checks if collection exists before indexing — saves API costs on re-runs
* Queries via `VectorStoreIndex.as_query_engine()`

**Key concepts learned:**

* Why `file_extractor={".pdf": PDFReader()}` is required (SimpleDirectoryReader merges PDF pages by default)
* Qdrant persistence — vectors survive process restart
* Collection existence check pattern

---

## P3 — Multi-doc + Metadata Filtering (`rag_multidoc.py`)

Production-ready multi-document RAG with metadata pre-filtering.

**What it does:**

* Indexes multiple documents from `docs/` directory with folder-based type detection
* Assigns `doc_type` per document based on parent folder name: `laws/` → `law`, `orders/` → `order`, `decrees/` → `decree`
* Uses `rglob` for recursive search across subfolders
* Pre-filtering: narrows Qdrant search space BEFORE vector comparison
* Hash registry (`doc_registry.json`): tracks SHA256 of each file, re-indexes only new/changed files
* Supports incremental indexing with `index.insert()`

**Document structure:**
```
docs/
  laws/     → doc_type="law"
  orders/   → doc_type="order"
  decrees/  → doc_type="decree"
```

**Key concepts learned:**

* Metadata stored alongside vectors in Qdrant payload
* Pre-filtering vs post-filtering (pre is faster and more accurate)
* Incremental indexing without full collection rebuild
* Hash-based change detection for large document sets
* Folder-based doc_type is more scalable than filename keyword matching

---

## P4 — Streamlit UI + 2FA (`streamlit_app.py`)

Web interface for the RAG pipeline with two-factor authentication.

**What it does:**

* Login form: password + TOTP code (Google Authenticator compatible)
* After auth: full RAG UI with document type filter, question input, answer display
* Sources panel: shows filename, doc\_type, similarity score, relevant text snippet
* Session-based auth: re-login required when browser tab is closed

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

* Runs as systemd service
* nginx reverse proxy with `^~` location prefix match
* `--server.baseUrlPath /rag` + `proxy_pass` without trailing slash

**Key concepts learned:**

* `@st.cache_resource` is critical — Streamlit re-runs entire script on every interaction
* WebSocket (`/_stcore/stream`) requires separate nginx location block
* `proxy_pass host:port` (no slash) vs `proxy_pass host:port/` (with slash) — different path handling
* TOTP: `valid_window=1` compensates for clock drift between server and phone

---

## P4.1 — Ukrainian Language Tuning (`streamlit_app.py`)

Targeted improvements for RAG quality on Ukrainian documents.

### Problem

`text-embedding-3-small` cosine similarity scores between Ukrainian questions and
document chunks were near-zero (0.01–0.02), causing irrelevant retrieval and hallucinations.

**Root cause:** semantic gap between question form and document form in Ukrainian
morphology — different word forms produce different vectors despite identical meaning.

### Solution 1 — HyDE (Hypothetical Document Embeddings)

Instead of embedding the raw question, GPT first generates a hypothetical answer,
then that answer is embedded for retrieval.

```
WITHOUT HyDE: question → embedding → score: 0.01
WITH HyDE:    question → GPT answer → embedding → score: 0.55–0.70
```

**Implementation** (manual HyDE, not via `TransformQueryEngine`):

```python
# Step 1: generate hypothetical answer in Ukrainian
hyde_prompt = f"Напиши коротку відповідь українською мовою на питання: {question}"
hypothetical = str(llm.complete(hyde_prompt))

# Step 2: use hypothetical answer as search query
response = query_engine.query(hypothetical)
```

### Solution 2 — Chunking optimization

`chunk_size=512, chunk_overlap=50` — one article per chunk for dense documents.

### Solution 3 — Morphological snippet extraction

Uses first 6 characters of query words as pseudo-stems for Ukrainian morphology matching.

### Results

| Metric | Before | After |
| --- | --- | --- |
| Similarity scores | 0.01–0.02 | 0.55–0.70 |
| Source relevance | Wrong documents | Correct documents |
| Answer language | English (GPT default) | Ukrainian |

---

## P4.2 — RAG Evaluation with RAGAS (`rag_eval.py`)

Quantitative evaluation of RAG system quality using the RAGAS framework.

### Why evaluation matters

"It seems to answer well" is not a quality metric. Before presenting a RAG system
to a client, you need concrete numbers that prove it works.

### Metrics

| Metric | What it measures | Needs ground truth |
| --- | --- | --- |
| **Faithfulness** | Does the answer come only from retrieved chunks? (hallucination detection) | No |
| **Answer Relevancy** | Is the answer on-topic for the question asked? | No |
| **Context Precision** | What fraction of retrieved chunks are actually useful? | No |
| **Context Recall** | Were all necessary chunks found? | Yes |

### Results on Ukrainian documents (similarity_top_k=3)

```
Faithfulness:      0.7889  ← system rarely hallucinates
Answer Relevancy:  0.7061  ← answers are on-topic
Context Precision: 0.6833  ← ~2 of 3 retrieved chunks are useful
Context Recall:    0.4000  ← limited by eval dataset quality, not the system
```

### Key finding: top_k trade-off

| similarity_top_k | Faithfulness | Answer Relevancy |
| --- | --- | --- |
| 3 | 0.79 | 0.71 |
| 5 | **0.93** | 0.61 ↓ |

More context reduces hallucinations but makes answers less focused.
`top_k=3` gives the best balance for this system.

### Technical notes (RAGAS 0.4.x)

* New API (`ragas.metrics.collections`) is incompatible with `evaluate()` — use old API
* `from ragas.metrics._faithfulness import faithfulness` — works with `evaluate()`
* `LangchainLLMWrapper` + `LangchainEmbeddingsWrapper` — required for compatibility
* `pip install langchain-openai` — needed for the wrapper

---

## P5 — AI Agent with RAG Tools (`agent_react.py`)

Multi-tool AI agent that autonomously decides which tools to use and in what order.

### Agent vs RAG

| | RAG | Agent |
| --- | --- | --- |
| Flow | Fixed: retrieve → generate | Dynamic: think → act → observe → repeat |
| Steps | Always 1 retrieval | As many as needed |
| Tools | 1 (vector search) | Multiple (search, filter, calculate, API...) |
| Use case | "What does this document say?" | "Find X, then calculate Y based on it" |

### Tools implemented

| Tool | Description |
| --- | --- |
| `search_documents` | General search across all documents |
| `search_in_laws` | Filtered search in laws only — for specific facts and numbers |
| `search_by_doc_type` | Search filtered by any doc_type |
| `calculate` | Safe mathematical expression evaluator |

### Example reasoning trace

```
Question: "How many employees and what is the monthly salary fund at 25000 UAH/month?"

→ search_in_laws("personnel count")
  Observation: "2993 persons" [law document, score: 0.821]

→ calculate("2993 * 25000")
  Observation: "2993 * 25000 = 74825000"

→ Final Answer: Personnel count is 2993. Monthly salary fund: 74,825,000 UAH.
```

### Key technical lessons

* `ReActAgent.from_tools()` no longer exists in newer LlamaIndex → use `FunctionAgent`
* `FunctionAgent.run()` is async → requires `async def main()` + `asyncio.run(main())`
* `allow_parallel_tool_calls=False` — critical for sequential reasoning
* Docstring of each function = instruction for LLM — quality of docstring determines quality of tool selection
* Short chunks have low cosine similarity even for exact content matches → dedicated filtered tool solves this
* `chat_history=None` between `run()` calls — no memory across questions (solved in P6)

### Why the docstring matters

The LLM never sees your Python code. It only sees the function name and docstring.
This is the entire "API contract" between your code and the model:

```python
def search_in_laws(query: str) -> str:
    """
    Search specifically in laws.
    Use this tool when you need specific facts: numbers, counts, dates.
    """
```

If the docstring is vague, the agent picks the wrong tool.
Treat docstrings as prompt engineering, not documentation.

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
            llama-index-embeddings-openai llama-index-readers-file \
            qdrant-client python-dotenv streamlit pyotp qrcode[pil] \
            ragas langchain-openai datasets

# Configure
cp .env.example .env
# Edit .env: add OPENAI_API_KEY, RAG_PASSWORD, RAG_TOTP_SECRET

# Run Qdrant (Docker)
docker run -d -p 6333:6333 qdrant/qdrant

# Add documents to docs/ subfolders
mkdir -p docs/laws docs/orders docs/decrees
# Copy your PDFs into appropriate subfolders

# Index documents
python rag_multidoc.py

# Run UI
streamlit run streamlit_app.py --server.baseUrlPath /rag

# Run agent (console)
python agent_react.py
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
eval_dataset.json
eval_results.json
totp_qr.png
__pycache__/
venv/
```
