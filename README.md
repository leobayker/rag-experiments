# rag-experiments

Learning path: RAG from scratch → LlamaIndex + Qdrant production stack.

Each script is a standalone, runnable experiment. No frameworks in early stages — everything built manually to understand what happens under the hood before abstracting it away.

---

## Roadmap

| File | Stage | Stack | Status |
|---|---|---|---|
| `rag_manual.py` | P1 — Manual RAG | pypdf · OpenAI Embeddings · NumPy · GPT-4o-mini | ✅ Done |
| `rag_llamaindex_qdrant.py` | P2 — Framework RAG | LlamaIndex · Qdrant · OpenAI | ⬜ Next |
| `rag_multidoc.py` | P3 — Multi-doc + metadata | LlamaIndex · Qdrant | ⬜ |
| `streamlit_app.py` | P4 — UI | Streamlit · LlamaIndex · Qdrant | ⬜ |
| `agent_react.py` | P5–P6 — ReAct Agent | LlamaIndex Agents · Tools | ⬜ |
| `agent_memory.py` | P7 — Agent with memory | LlamaIndex · Redis/Qdrant | ⬜ |

---

## P1 — Manual RAG (`rag_manual.py`)

RAG pipeline built without any framework. Every component is explicit.

**Pipeline:**
```
PDF → text extraction → word-based chunking (500w / 50w overlap)
    → OpenAI embeddings (text-embedding-3-small)
    → cosine similarity search (NumPy)
    → GPT-4o-mini answer with retrieved context
```

**What this demonstrates:**
- How embeddings represent semantic meaning as vectors
- How cosine similarity finds relevant chunks (~0.7+ = relevant, <0.4 = unrelated)
- The full RAG loop: index → retrieve → augment → generate

**Setup:**
```bash
pip install pypdf openai numpy
export OPENAI_API_KEY=sk-...
```

**Run:**
```bash
# Place your PDF as test.pdf in the same directory
python rag_manual.py
```

**Example output:**
```
Reading PDF...
Splitting into chunks...
Total chunks: 47
Generating embeddings... (this may take a moment)
Ready. Ask your questions (Ctrl+C to exit):

Question: What is the main topic of the document?
Scores: [0.847, 0.821, 0.793]

Answer: Based on the context provided...
```

---

## Environment

All experiments run on a self-hosted VPS (Ubuntu 22.04). No local GPU required for P1–P4 — OpenAI API handles LLM and embeddings.

Starting from P9, experiments move to fully on-premise stack with Ollama (no external API calls).

---

## Related repos

- [`network-infrastructure`](https://github.com/leobayker/network-infrastructure) — enterprise network architecture
- [`mikrotik-configs`](https://github.com/leobayker/mikrotik-configs) — RouterOS configs & automation
- [`automation-scripts`](https://github.com/leobayker/automation-scripts) — Python/PowerShell/Bash scripts
