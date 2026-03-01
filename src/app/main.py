"""Streamlit UI for RAG."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from config.settings import (
    FAISS_INDEX_PATH,
    CHUNKS_METADATA_PATH,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)
from src.embeddings import get_encoder
from src.llm.ollama_client import OllamaClient, OllamaError
from src.rag.pipeline import RAGPipeline
from src.vector_store.faiss_store import FAISSStore


def _load_rag() -> RAGPipeline | None:
    """Load encoder, FAISS, Ollama; return RAG pipeline."""
    if not FAISS_INDEX_PATH.exists():
        return None
    encoder = get_encoder(EMBEDDING_MODEL)
    store = FAISSStore(dimension=EMBEDDING_DIM)
    store.load(FAISS_INDEX_PATH, CHUNKS_METADATA_PATH)
    llm = OllamaClient(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
    rag = RAGPipeline(encoder=encoder, store=store, llm=llm, top_k=5)
    # Ollama warm-up: first generate is slow; warm up at startup.
    try:
        llm.generate("OK", system="Reply with one word.", temperature=0)
    except Exception:
        pass
    return rag


def _render_result(result: dict) -> None:
    """Render answer and expandable context with page numbers."""
    answer = result.get("answer", "")
    sources = result.get("sources", [])
    scores = result.get("similarity_scores", [])

    st.markdown("### Answer")
    st.markdown(answer if answer else "_No answer._")
    st.divider()

    if not sources:
        return

    with st.expander("**Context** — retrieved chunks", expanded=False):
        for i, src in enumerate(sources):
            page = src.get("page_number")
            text = (src.get("text") or "").strip()
            score = scores[i] if i < len(scores) else None
            path_label = src.get("source_path") or "—"
            path_short = Path(path_label).name if path_label != "—" else "—"

            header_parts = [f"Page {page}"] if page is not None else []
            if score is not None:
                header_parts.append(f"similarity: **{score:.3f}**")
            header = " · ".join(header_parts)

            st.caption(header)
            st.caption(f"Source: `{path_short}`")
            st.text(text[:500] + ("..." if len(text) > 500 else ""))
            if i < len(sources) - 1:
                st.divider()


def main() -> None:
    st.set_page_config(
        page_title="RAG — Local",
        page_icon="📚",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    st.title("RAG")
    st.caption("Ask questions over your documents. Answers use only the retrieved context.")

    rag = _load_rag()
    if rag is None:
        st.warning(
            "No index found. Add PDFs to `data/raw_documents/` and run "
            "`python scripts/build_index.py`, then restart the app."
        )
        st.stop()

    with st.form("ask_form", clear_on_submit=False):
        query = st.text_input(
            "Question",
            placeholder="e.g. What was the revenue in Q3?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Ask")

    if submitted and query and query.strip():
        with st.spinner("Searching and generating..."):
            try:
                result = rag.run(query.strip(), temperature=0.1)
                _render_result(result)
            except OllamaError as e:
                st.error(f"Ollama error: {e}")
    elif submitted and not (query or "").strip():
        st.info("Enter a question and click **Ask**.")

    with st.sidebar:
        st.header("Settings")
        st.caption("Config: `config/settings.py`")
        st.divider()
        st.caption(f"Model: **{OLLAMA_MODEL}**")
        st.caption(f"Embeddings: **{EMBEDDING_MODEL}**")


if __name__ == "__main__":
    main()
