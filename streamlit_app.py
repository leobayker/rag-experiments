"""
P4 — Streamlit UI for RAG (Multi-doc + Metadata Filtering)
With two-factor authentication (password + TOTP via Google Authenticator)

Stack: LlamaIndex + Qdrant + OpenAI + pyotp
Collection: p3_multidoc (created in P3)
"""

import streamlit as st
from dotenv import load_dotenv
import os
import pyotp

# Load .env from current directory (OPENAI_API_KEY, RAG_PASSWORD, RAG_TOTP_SECRET)
load_dotenv()

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator
from qdrant_client import QdrantClient

# ─── Config ───────────────────────────────────────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "p3_multidoc")
RAG_PASSWORD = os.getenv("RAG_PASSWORD", "")
TOTP_SECRET = os.getenv("RAG_TOTP_SECRET", "")

# ─── Authentication ───────────────────────────────────────────────────────────
def check_auth(password: str, totp_code: str) -> tuple[bool, str]:
    """
    Verifies password and TOTP code.
    Returns (True, "") if valid, or (False, "error message").

    Error message is intentionally generic — attacker should not know
    which factor failed.
    """
    if password != RAG_PASSWORD:
        return False, "Invalid credentials"

    totp = pyotp.TOTP(TOTP_SECRET)
    # valid_window=1 accepts codes from ±30 seconds around current time
    # compensates for clock drift between server and phone
    if not totp.verify(totp_code, valid_window=1):
        return False, "Invalid credentials"

    return True, ""


def show_login():
    """Renders login form."""
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
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error(msg)


# ─── RAG Index ────────────────────────────────────────────────────────────────
@st.cache_resource
def get_index():
    """
    Initializes Qdrant client and LlamaIndex VectorStoreIndex.

    @st.cache_resource caches the object across all reruns —
    without it, Streamlit would recreate the connection on every
    button click or text input event.
    """
    client = QdrantClient(url=QDRANT_URL)
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )
    return index


# ─── RAG Query ────────────────────────────────────────────────────────────────
def query_rag(index, question: str, doc_type_filter: str = "All"):
    """
    Runs RAG query. Applies metadata pre-filter if doc_type_filter is set.
    Returns (answer, list of sources).
    """
    if doc_type_filter != "All":
        filters = MetadataFilters(filters=[
            MetadataFilter(key="doc_type", value=doc_type_filter, operator=FilterOperator.EQ)
        ])
        query_engine = index.as_query_engine(similarity_top_k=5, filters=filters)
    else:
        query_engine = index.as_query_engine(similarity_top_k=5)

    response = query_engine.query(question)

    sources = []
    if hasattr(response, "source_nodes"):
        for node in response.source_nodes:
            meta = node.metadata
            sources.append({
                # LlamaIndex stores filename as "file_name" (not "filename")
                "filename": meta.get("file_name", "unknown"),
                "doc_type": meta.get("doc_type", "unknown"),
                "score": round(node.score, 3) if node.score else "n/a",
                "text_preview": node.text[:200] + "..." if len(node.text) > 200 else node.text,
            })

    return str(response), sources


# ─── Main UI ──────────────────────────────────────────────────────────────────
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
        st.markdown(f"**Collection:** `{COLLECTION_NAME}`")
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
            placeholder="e.g. What is the password policy? / How to handle incidents?",
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
                        st.text(src["text_preview"])

    with col2:
        st.markdown("### 💡 Example questions")
        examples = [
            "What is the password policy?",
            "How to handle security incidents?",
            "What are the backup procedures?",
            "Who is responsible for access management?",
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
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        show_login()
    else:
        show_main()


if __name__ == "__main__":
    main()
