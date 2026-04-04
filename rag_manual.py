"""
RAG from scratch — no frameworks, pure Python.

Pipeline:
  PDF → text extraction → chunking → embeddings (OpenAI) → cosine similarity → GPT answer

Requirements:
    pip install pypdf openai numpy

Environment:
    export OPENAI_API_KEY=sk-...
"""

import numpy as np
from pypdf import PdfReader
from openai import OpenAI

client = OpenAI()


def extract_text(pdf_path: str) -> str:
    """Extract full text from a PDF file."""
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() for page in reader.pages)


def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping word-based chunks.

    chunk_size: number of words per chunk
    overlap:    number of words shared between consecutive chunks
    """
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def get_embedding(text: str) -> np.ndarray:
    """Get embedding vector for a text string using OpenAI API."""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return np.array(response.data[0].embedding)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors. Range: -1 to 1."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def find_top_chunks(
    query: str,
    chunk_embeddings: list[np.ndarray],
    chunks: list[str],
    top_n: int = 3
) -> tuple[list[str], list[float]]:
    """Find top-N most relevant chunks for a query using cosine similarity."""
    query_emb = get_embedding(query)
    scores = [cosine_similarity(query_emb, emb) for emb in chunk_embeddings]
    top_indices = np.argsort(scores)[::-1][:top_n]
    return [chunks[i] for i in top_indices], [scores[i] for i in top_indices]


def answer_with_context(query: str, context_chunks: list[str]) -> str:
    """Send query + retrieved context to GPT and return the answer."""
    context = "\n\n---\n\n".join(context_chunks)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Answer only based on the provided context. If the answer is not in the context, say so."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ]
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print("Reading PDF...")
    text = extract_text("test.pdf")

    print("Splitting into chunks...")
    chunks = split_into_chunks(text, chunk_size=500, overlap=50)
    print(f"Total chunks: {len(chunks)}")

    print("Generating embeddings... (this may take a moment)")
    chunk_embeddings = [get_embedding(chunk) for chunk in chunks]

    print("Ready. Ask your questions (Ctrl+C to exit):\n")
    while True:
        query = input("Question: ")
        if not query:
            continue
        top_chunks, scores = find_top_chunks(query, chunk_embeddings, chunks)
        print(f"Scores: {[round(s, 3) for s in scores]}")
        answer = answer_with_context(query, top_chunks)
        print(f"\nAnswer: {answer}\n")
