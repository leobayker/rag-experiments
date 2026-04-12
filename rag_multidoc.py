"""
P3 — Multi-document RAG with Metadata Filtering

New compared to P2:
- Hash-based registry: tracks which files are already indexed
- Explicit metadata per document (doc_type, file_name, source)
- Metadata filtering at query time: search only in specific doc types
"""

import os
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters, FilterOperator
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
import qdrant_client

load_dotenv()

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

DOCS_DIR = "/opt/rag-experiments/docs"

# Відповідність назви папки → doc_type
FOLDER_TYPE_MAP = {
    "laws": "law",
    "orders": "order",
    "decrees": "decree",
    "general": "general",
}
REGISTRY_PATH = "/opt/rag-experiments/doc_registry.json"  # hash registry stored here
COLLECTION_NAME = "p3_multidoc"                            # separate collection from P2
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333


# ─────────────────────────────────────────────
# HASH UTILS
#
# We compute SHA256 hash of each file.
# If the hash hasn't changed since last run → file hasn't changed → skip indexing.
# SHA256 produces a 64-char string, practically unique per file content.
# We read in 8KB chunks to avoid loading large files entirely into RAM.
# ─────────────────────────────────────────────

def get_file_hash(filepath: str) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


# ─────────────────────────────────────────────
# HASH REGISTRY
#
# A simple JSON file on disk: { "filepath": "hash", ... }
# Persists between script runs so we remember what was already indexed.
# On first run the file doesn't exist → return empty dict → index everything.
# After indexing each file → write its hash to the registry immediately,
# so a crash mid-run won't lose progress on already-indexed files.
# ─────────────────────────────────────────────

def load_registry() -> dict:
    if Path(REGISTRY_PATH).exists():
        with open(REGISTRY_PATH, "r") as f:
            return json.load(f)
    return {}


def save_registry(registry: dict):
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


# ─────────────────────────────────────────────
# METADATA ASSIGNMENT
#
# We determine doc_type by keywords in the filename.
# This is the simplest approach — good for demo and small projects.
# In production: use folder structure, a CSV manifest, or AI-based classification.
#
# Metadata dict is attached to every chunk of this document in Qdrant.
# Later we can filter by any of these fields at query time.
# ─────────────────────────────────────────────

def get_doc_metadata(filepath: str) -> dict:
    # Визначаємо тип по назві папки в якій лежить файл
    # Path(filepath).parent.name — назва безпосередньої батьківської папки
    folder = Path(filepath).parent.name
    doc_type = FOLDER_TYPE_MAP.get(folder, "general")

    return {
        "file_name": Path(filepath).name,
        "doc_type": doc_type,
        "source": "local_docs",
    }


# ─────────────────────────────────────────────
# LLM AND EMBEDDING SETTINGS
#
# gpt-4o-mini: cheap and fast, good enough for Q&A over structured docs.
# temperature=0: deterministic answers — important for factual RAG.
# text-embedding-3-small: 1536-dim vectors, best price/quality ratio from OpenAI.
# ─────────────────────────────────────────────

def init_settings():
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)


# ─────────────────────────────────────────────
# INDEX INITIALIZATION
#
# We connect to Qdrant and get or create the index.
#
# from_vector_store(): connects to an existing collection without re-indexing.
# VectorStoreIndex(nodes=[]): creates a new empty collection.
#
# Why check collection existence manually:
# LlamaIndex doesn't expose a clean "get or create" API,
# so we query Qdrant directly to decide which path to take.
# ─────────────────────────────────────────────

def get_index() -> VectorStoreIndex:
    client = qdrant_client.QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    collections = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in collections:
        print(f"  Collection '{COLLECTION_NAME}' found — connecting.")
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
        )
    else:
        print(f"  Collection '{COLLECTION_NAME}' not found — creating.")
        index = VectorStoreIndex(nodes=[], storage_context=storage_context)

    return index


# ─────────────────────────────────────────────
# INCREMENTAL INDEXING
#
# Core logic of P3 — only index files that are new or changed.
#
# Flow per file:
#   1. Compute SHA256 hash
#   2. Check registry: hash matches → SKIP, hash missing/different → INDEX
#   3. SimpleDirectoryReader reads the file
#   4. file_metadata callback attaches our metadata to every chunk
#   5. index.insert(doc) adds chunks to Qdrant WITHOUT rebuilding the collection
#   6. Update registry with new hash immediately after successful indexing
#
# index.insert() is the key difference from P2:
# in P2 we rebuilt the whole index every time.
# Here we surgically add only new documents — no wasted API calls.
# ─────────────────────────────────────────────

def index_new_documents(index: VectorStoreIndex):
    registry = load_registry()
    docs_path = Path(DOCS_DIR)
    all_files = list(docs_path.rglob("*.pdf")) + list(docs_path.rglob("*.txt"))

    new_files = []
    for filepath in all_files:
        file_hash = get_file_hash(str(filepath))
        if str(filepath) in registry and registry[str(filepath)] == file_hash:
            print(f"  [SKIP] {filepath.name} — already indexed")
        else:
            print(f"  [NEW]  {filepath.name} — will be indexed")
            new_files.append(filepath)

    if not new_files:
        print("  No new files.")
        return

    for filepath in new_files:
        metadata = get_doc_metadata(str(filepath))
        print(f"  Indexing: {filepath.name} (doc_type: {metadata['doc_type']})")

        # file_metadata is a callback: called once per file, returns metadata dict.
        # LlamaIndex attaches this dict to every chunk produced from this file.
        reader = SimpleDirectoryReader(
            input_files=[str(filepath)],
            file_metadata=lambda path: get_doc_metadata(path),
        )
        documents = reader.load_data()

        for doc in documents:
            index.insert(doc)

        # save hash right after indexing this file
        # if script crashes on next file, this one won't be re-indexed
        registry[str(filepath)] = get_file_hash(str(filepath))
        save_registry(registry)
        print(f"  ✓ {filepath.name} done")


# ─────────────────────────────────────────────
# QUERY WITH METADATA FILTERING
#
# MetadataFilter: a condition on a single metadata field.
#   key="doc_type", value="policy", operator=EQ
#   → only chunks where doc_type == "policy" are considered
#
# MetadataFilters: wrapper that can hold multiple filters with AND/OR logic.
#
# How filtering works internally:
#   Qdrant applies the filter BEFORE vector search.
#   It doesn't search all vectors and then filter results —
#   it restricts the search space first, then finds nearest neighbors within it.
#   This is faster and more precise than post-filtering.
#
# filter_type=None → no filter → search across all documents.
# ─────────────────────────────────────────────

def query_with_filter(index: VectorStoreIndex, question: str, filter_type: str = None):
    if filter_type:
        filters = MetadataFilters(filters=[
            MetadataFilter(key="doc_type", value=filter_type, operator=FilterOperator.EQ)
        ])
        query_engine = index.as_query_engine(similarity_top_k=3, filters=filters)
        print(f"\n[Filter active: doc_type='{filter_type}']\n")
    else:
        query_engine = index.as_query_engine(similarity_top_k=3)
        print(f"\n[No filter — searching all documents]\n")

    response = query_engine.query(question)
    return response


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("P3 — Multi-doc RAG with Metadata Filtering")
    print("=" * 60)

    print("\n[1/3] Initializing models...")
    init_settings()

    print("\n[2/3] Connecting to Qdrant...")
    index = get_index()

    print("\n[3/3] Checking for new documents...")
    index_new_documents(index)

    print("\n" + "=" * 60)
    print("Ready!\n")
    print("Commands:")
    print("  /filter policy       → search only in policy docs")
    print("  /filter procedure    → search only in procedure docs")
    print("  /filter report       → search only in report docs")
    print("  /filter off          → remove filter")
    print("  /quit                → exit")
    print("=" * 60 + "\n")

    active_filter = None

    while True:
        user_input = input("Question: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "/quit":
            print("Exiting.")
            break
        if user_input.lower() == "/filter off":
            active_filter = None
            print("[Filter removed — searching all documents]\n")
            continue
        if user_input.lower().startswith("/filter "):
            active_filter = user_input[8:].strip()
            print(f"[Filter set: doc_type='{active_filter}']\n")
            continue
        try:
            response = query_with_filter(index, user_input, active_filter)
            print(f"Answer: {response}\n")

            # show which chunks were used to generate the answer
            # score = cosine similarity: 1.0 = identical, 0.0 = unrelated
            if hasattr(response, 'source_nodes') and response.source_nodes:
                print("Sources:")
                for node in response.source_nodes:
                    doc_type = node.metadata.get('doc_type', 'unknown')
                    file_name = node.metadata.get('file_name', 'unknown')
                    score = round(node.score, 3) if node.score else 'N/A'
                    print(f"  [{doc_type}] {file_name} (score: {score})")
                print()
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
