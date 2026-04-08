"""
P4 — Streamlit UI для RAG (Multi-doc + Metadata)
З двофакторною автентифікацією (пароль + TOTP)
"""

import streamlit as st
from dotenv import load_dotenv
import os
import pyotp

load_dotenv("/opt/rag-experiments/.env")

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.indices.query.query_transform.base import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator
from qdrant_client import QdrantClient

# ─── Константи ────────────────────────────────────────────────────────────────
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "p3_multidoc"
RAG_PASSWORD = os.getenv("RAG_PASSWORD", "")
TOTP_SECRET = os.getenv("RAG_TOTP_SECRET", "")

# ─── Автентифікація ───────────────────────────────────────────────────────────
def check_auth(password: str, totp_code: str) -> tuple[bool, str]:
    """
    Перевіряє пароль і TOTP код.
    Повертає (True, "") якщо все ОК, або (False, "повідомлення про помилку").
    
    Перевірка навмисно не розділяє помилки пароля і TOTP —
    щоб атакуючий не знав який саме фактор невірний.
    """
    if password != RAG_PASSWORD:
        return False, "Invalid credentials"
    
    totp = pyotp.TOTP(TOTP_SECRET)
    # valid_window=1 — приймає код з попереднього і наступного 30-секундного вікна
    # це компенсує невеликий drift годинника між сервером і телефоном
    if not totp.verify(totp_code, valid_window=1):
        return False, "Invalid credentials"
    
    return True, ""

def show_login():
    """Показує форму логіну."""
    st.set_page_config(page_title="RAG Demo — Login", page_icon="🔐", layout="centered")
    
    st.title("🔐 RAG Document Assistant")
    st.caption("Enter your credentials to continue")
    
    with st.form("login_form"):
        password = st.text_input("Password", type="password")
        totp_code = st.text_input(
            "2FA Code",
            placeholder="6-digit code from Google Authenticator",
            max_chars=6,
        )
        submitted = st.form_submit_button("Login", type="primary", use_container_width=True)
        
        if submitted:
            if not password or not totp_code:
                st.error("Please fill in all fields")
            else:
                ok, msg = check_auth(password, totp_code)
                if ok:
                    # Зберігаємо стан авторизації в session_state
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error(msg)

# ─── Ініціалізація RAG ────────────────────────────────────────────────────────
@st.cache_resource
def get_index():
    Settings.llm = OpenAI(
        model="gpt-4o-mini",
        temperature=0,
        system_prompt="Ти асистент який відповідає ВИКЛЮЧНО українською мовою. Використовуй лише інформацію з наданого контексту."
    )
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    client = QdrantClient(url=QDRANT_URL)
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )
    return index

# ─── RAG запит ────────────────────────────────────────────────────────────────
def query_rag(index, question: str, doc_type_filter: str = "All"):
    if doc_type_filter != "All":
        filters = MetadataFilters(filters=[
            MetadataFilter(key="doc_type", value=doc_type_filter, operator=FilterOperator.EQ)
        ])
        query_engine = index.as_query_engine(similarity_top_k=5, filters=filters)
    else:
        query_engine = index.as_query_engine(similarity_top_k=5)

    # Ручний HyDE: генеруємо гіпотетичну відповідь → шукаємо по ній
    llm = Settings.llm
    hyde_prompt = f"Напиши коротку відповідь українською мовою на питання: {question}"
    hypothetical = str(llm.complete(hyde_prompt))
    # Пошук по гіпотетичній відповіді — source_nodes будуть точними
    response = query_engine.query(hypothetical)
    sources = []
    if hasattr(response, "source_nodes"):
        for node in response.source_nodes:
            if node.score is None or node.score < 0:
                continue
            meta = node.metadata
            sources.append({
                "filename": meta.get("file_name", "unknown"),
                "doc_type": meta.get("doc_type", "unknown"),
                "score": round(node.score, 3),
                "text_preview": node.text,
            })
    return str(response), sources

# ─── Допоміжна функція: витягує релевантний фрагмент з chunk ─────────────────
def extract_relevant_snippet(text: str, question: str, context_sentences: int = 2) -> str:
    # Розбиваємо на речення
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 20]
    if not sentences:
        return text[:400]
    # Ключові слова з питання (слова довші за 3 символи)
    # Використовуємо корені слів (перші 6 символів) для морфологічного співпадіння
    keywords = [w.lower()[:6] for w in question.split() if len(w) > 4]
    # Рахуємо скільки ключових слів є в кожному реченні
    scores = []
    for s in sentences:
        score = sum(1 for kw in keywords if kw in s.lower())
        scores.append(score)
    # Знаходимо речення з найвищим score
    best_idx = scores.index(max(scores))
    # Беремо контекст навколо: N речень до і після
    start = max(0, best_idx - context_sentences)
    end = min(len(sentences), best_idx + context_sentences + 1)
    snippet = ". ".join(sentences[start:end]) + "."
    # Якщо найкращий score = 0 — показуємо початок chunk
    if max(scores) == 0:
        return text[:400] + ("..." if len(text) > 400 else "")
    return snippet

# ─── Основний UI ──────────────────────────────────────────────────────────────
def show_main():
    st.set_page_config(
        page_title="RAG Demo",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 RAG Document Assistant")
    st.caption("Powered by LlamaIndex + Qdrant + OpenAI | P4 Demo")

    with st.sidebar:
        st.header("⚙️ Settings")
        doc_type = st.selectbox(
            "Filter by document type",
            options=["All", "policy", "procedure", "report", "manual"],
            index=0,
        )
        st.markdown("---")
        st.markdown("**Collection:** `p3_multidoc`")
        st.markdown("**Embedding:** text-embedding-3-small")
        st.markdown("**LLM:** gpt-4o-mini")
        st.markdown("---")
        if st.button("🚪 Logout"):
            st.session_state["authenticated"] = False
            st.rerun()

    with st.spinner("Connecting to Qdrant..."):
        index = get_index()

    col1, col2 = st.columns([2, 1])

    with col1:
        question = st.text_area(
            "Ask a question:",
            placeholder="e.g. Що таке УДО України",
            height=100,
        )

        if st.button("🔍 Search", type="primary", disabled=not question.strip()):
            with st.spinner("Searching..."):
                answer, sources = query_rag(index, question, doc_type)

            st.markdown("### 💬 Answer")
            st.markdown(answer)

            if sources:
                st.markdown("### 📎 Sources")
                for i, src in enumerate(sources, 1):
                    with st.expander(f"[{i}] {src['filename']} ({src['doc_type']}) — score: {src['score']}"):
                        st.markdown(f"**Doc type:** `{src['doc_type']}`")
                        st.markdown(f"**File:** `{src['filename']}`")
                        st.markdown(f"**Similarity score:** `{src['score']}`")
                        st.markdown("**Preview:**")
                        snippet = extract_relevant_snippet(src["text_preview"], question)
                        st.markdown(f"> {snippet}")

    with col2:
        st.markdown("### 💡 Приклади запитань")
        examples = [
            "Чим займається УДО України?",
            "Кому підпорядковане УДО України?",
            "Чому УДО України має право обмежувати прохід людей?",
        ]
        for ex in examples:
            if st.button(ex, key=ex):
                with st.spinner("Searching..."):
                    answer, sources = query_rag(index, ex, doc_type)
                st.markdown(f"**Question:** {ex}")
                st.markdown("### 💬 Answer")
                st.markdown(answer)
                if sources:
                    for i, src in enumerate(sources, 1):
                        with st.expander(f"[{i}] {src['filename']} — score: {src['score']}"):
                            st.text(src["text_preview"])

# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    # Ініціалізуємо стан авторизації якщо його ще немає
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        show_login()
    else:
        show_main()

if __name__ == "__main__":
    main()
