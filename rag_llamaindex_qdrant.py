import os
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

from qdrant_client import QdrantClient

# -------------------------------------------------------
# BLOCK 1 — Load API key from .env file
# load_dotenv reads .env and adds variables to os.environ
# LlamaIndex and OpenAI SDK automatically pick up
# OPENAI_API_KEY from os.environ — no need to pass it manually
# -------------------------------------------------------
load_dotenv()  # reads .env from current working directory

# -------------------------------------------------------
# BLOCK 2 — Global LlamaIndex settings
# Settings is a singleton config that applies to the whole script
# text-embedding-3-small: newer model, ~5x cheaper than ada-002
# quality for RAG tasks is equal or better
# -------------------------------------------------------
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# -------------------------------------------------------
# BLOCK 3 — Connect to Qdrant
# Qdrant is running in Docker on this same VPS, port 6333
# This only creates the client object — no actual connection yet
# Real connection happens on the first request
# -------------------------------------------------------
client = QdrantClient(host="localhost", port=6333)

# -------------------------------------------------------
# BLOCK 4 — Vector Store and Storage Context
# QdrantVectorStore: adapter/bridge between LlamaIndex and Qdrant
# collection_name="p2_documents": collection in Qdrant (like a table in DB)
# If collection doesn't exist — Qdrant creates it automatically
#
# StorageContext tells LlamaIndex where to write vectors
# from_defaults(vector_store=...): use Qdrant for vectors,
# use defaults for everything else
# -------------------------------------------------------
vector_store = QdrantVectorStore(client=client, collection_name="p2_documents")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# -------------------------------------------------------
# BLOCK 5 — Load documents
# SimpleDirectoryReader reads all files from docs/ folder — scalable,
# works with multiple files and formats (PDF, TXT, DOCX)
#
# CHANGED vs default behavior:
# Default SimpleDirectoryReader merges all PDF pages into 1 Document
# → results in 1 chunk in Qdrant → all queries return same chunk
# Fix: file_extractor={".pdf": PDFReader()} tells LlamaIndex to use
# PDFReader specifically for .pdf files — it splits by page
# → each page = separate Document = diverse chunks = proper retrieval
#
# Scalable: add more files to docs/ folder — they load automatically
# Other formats (TXT, DOCX) use default readers, unaffected
from llama_index.readers.file import PDFReader

print("Loading documents...")
documents = SimpleDirectoryReader(
    "docs",
    file_extractor={".pdf": PDFReader()}
).load_data()
print(f"Loaded pages/sections: {len(documents)}")
# -------------------------------------------------------
# BLOCK 6 — Indexing with persistence check
# CHANGED vs naive approach:
# Naive: delete collection → re-index every run → wastes OpenAI API money
# and time. Wrong approach for production.
#
# Correct approach:
# Check if collection already exists in Qdrant and has vectors
# If yes — skip indexing, just load the existing index
# If no — index documents and store vectors in Qdrant
#
# This means: first run takes a few seconds (API calls for embeddings)
# All subsequent runs are instant — Qdrant already has the vectors
# Re-index only when documents actually change
collections = [c.name for c in client.get_collections().collections]

if "p2_documents" in collections:
    print("Collection 'p2_documents' already exists — skipping indexing.")
    index = VectorStoreIndex.from_vector_store(vector_store)
else:
    print("Indexing documents into Qdrant...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )
    print("Done. Collection 'p2_documents' is populated in Qdrant.\n")
# -------------------------------------------------------
# BLOCK 7 — Query Engine
# as_query_engine() creates the object that handles queries
# similarity_top_k=3: retrieve 3 most relevant chunks per query
# (P1 used only 1 chunk — more context = better answers)
#
# What happens under the hood on each query():
# 1. Question → embedding vector via OpenAI API
# 2. Qdrant finds 3 nearest vectors (cosine similarity)
# 3. Those 3 chunks are passed to GPT as context
# 4. GPT generates answer grounded in that context
# -------------------------------------------------------
query_engine = index.as_query_engine(similarity_top_k=3)

# -------------------------------------------------------
# BLOCK 8 — Interactive query loop
# Instead of hardcoding a question, we ask the user to type it
# Loop runs until user types 'exit' or presses Ctrl+C
# This is a temporary solution before Streamlit UI in P4
# -------------------------------------------------------
print("\n--- RAG ready. Type your question (or 'exit' to quit) ---\n")

while True:
    question = input("Question: ").strip()

    # Exit condition
    if question.lower() in ("exit", "quit", "q"):
        print("Bye.")
        break

    # Skip empty input
    if not question:
        continue

    response = query_engine.query(question)
    print(f"\nAnswer:\n{response}\n")

    # Show source chunks with scores
    print("--- Source chunks ---")
    for i, node in enumerate(response.source_nodes):
        print(f"\nChunk {i+1} | score: {node.score:.4f}")
        print("-" * 40)
        print(node.text[:300], "...")
    print()
