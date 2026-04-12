"""
P4.2 — RAG Evaluation with RAGAS
Metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall
RAGAS version: 0.4.x
"""

import json
import os
from dotenv import load_dotenv

load_dotenv("/opt/rag-experiments/.env")

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from qdrant_client import QdrantClient

from ragas import evaluate
from ragas.metrics._faithfulness import faithfulness
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_precision
from ragas.metrics._context_recall import context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from datasets import Dataset

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "p3_multidoc"
EVAL_DATASET_PATH = "/opt/rag-experiments/eval_dataset.json"

# Global metrics list — ініціалізується в init()
metrics = []

# ─────────────────────────────────────────────
# INIT
# LlamaIndex settings для RAG запитів
# RAGAS 0.4.x: llm_factory/embedding_factory замість LangChain wrappers
# ─────────────────────────────────────────────

def init():
    global metrics

    # LlamaIndex — для RAG запитів
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    # Qdrant connection
    client = QdrantClient(url=QDRANT_URL)
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )

    # RAGAS використовує старий API метрик (сумісний з evaluate())
    # LangChain wrapper потрібен — старі метрики не підтримують llm_factory
    ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    faithfulness.llm = ragas_llm
    answer_relevancy.llm = ragas_llm
    answer_relevancy.embeddings = ragas_embeddings
    context_precision.llm = ragas_llm
    context_recall.llm = ragas_llm

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    return index

# ─────────────────────────────────────────────
# RAG QUERY
# Повертає відповідь + список контекстів (chunks)
# contexts — список рядків, саме такий формат очікує RAGAS
# ─────────────────────────────────────────────

def query_rag(index, question: str) -> tuple[str, list[str]]:
    query_engine = index.as_query_engine(similarity_top_k=5)
    response = query_engine.query(question)

    answer = str(response)
    contexts = [node.text for node in response.source_nodes]

    return answer, contexts

# ─────────────────────────────────────────────
# BUILD EVAL DATASET
# RAGAS очікує datasets.Dataset з колонками:
#   question     — питання
#   answer       — відповідь системи
#   contexts     — список chunks які система використала
#   ground_truth — еталонна відповідь (потрібна для context_recall)
# ─────────────────────────────────────────────

def build_ragas_dataset(index, eval_pairs: list[dict]) -> Dataset:
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    total = len(eval_pairs)
    for i, pair in enumerate(eval_pairs):
        question = pair["question"]
        ground_truth = pair["ground_truth"]

        print(f"  [{i+1}/{total}] {question[:60]}...")

        answer, ctx = query_rag(index, question)

        questions.append(question)
        answers.append(answer)
        contexts.append(ctx)
        ground_truths.append(ground_truth)

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("P4.2 — RAG Evaluation with RAGAS")
    print("=" * 60)

    print("\n[1/4] Initializing models and Qdrant connection...")
    index = init()

    print("\n[2/4] Loading eval dataset...")
    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        eval_pairs = json.load(f)
    print(f"  Loaded {len(eval_pairs)} question-answer pairs")

    print("\n[3/4] Running RAG queries to collect answers and contexts...")
    ragas_dataset = build_ragas_dataset(index, eval_pairs)

    print("\n[4/4] Running RAGAS evaluation...")
    print("  (This will make additional OpenAI API calls for LLM-as-judge)\n")

    result = evaluate(
        dataset=ragas_dataset,
        metrics=metrics,
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for metric_name in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        val = result[metric_name]
        if isinstance(val, list):
            import numpy as np
            val = float(np.nanmean(val))
        print(f"  {metric_name:<22} {val:.4f}")
    print("=" * 60)

    # Зберігаємо детальні результати по кожному питанню
    results_path = "/opt/rag-experiments/eval_results.json"
    result_df = result.to_pandas()
    result_df.to_json(results_path, orient="records", force_ascii=False, indent=2)
    print(f"\nDetailed results saved to: {results_path}")

if __name__ == "__main__":
    main()
