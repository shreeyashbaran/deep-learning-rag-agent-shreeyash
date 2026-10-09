"""Streamlit application for the Markdown-only Part 1 RAG workshop.

Run from the repository root:
    uv run streamlit run src/rag_agent/ui/app.py

Part 1 uses direct retrieval and generation. The starter LangGraph agent is
reserved for later workshop parts and is not invoked by this application.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import streamlit as st
from loguru import logger

from rag_agent.agent.state import IngestionResult
from rag_agent.config import LLMFactory, get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.qa import answer_question
from rag_agent.vectorstore.store import VectorStoreManager


# Cached resources survive Streamlit reruns.
@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    return VectorStoreManager()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    return DocumentChunker()


@st.cache_resource
def get_llm():
    """Configure the Groq client once, when the first question is submitted."""
    return LLMFactory().create()


def initialise_session_state() -> None:
    defaults = {
        "chat_history": [],
        "ingested_documents": [],
        "selected_document": None,
        "last_ingestion_result": None,
        "topic_filter": None,
        "difficulty_filter": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_ingestion_panel(store: VectorStoreManager, chunker: DocumentChunker) -> None:
    """Upload Markdown documents, report ingestion, and browse/remove sources."""
    st.sidebar.header("📂 Corpus Ingestion")
    files = st.sidebar.file_uploader(
        "Upload study notes (.md)", type=["md"], accept_multiple_files=True
    )
    if st.sidebar.button("Ingest Documents", disabled=not files):
        result = IngestionResult()
        with st.spinner("Chunking and embedding documents..."):
            with TemporaryDirectory(prefix="rag-upload-") as temporary:
                for upload in files:
                    name = Path(upload.name.replace("\\", "/")).name
                    try:
                        path = Path(temporary) / name
                        path.write_bytes(upload.getvalue())
                        chunks = chunker.chunk_file(path)
                        if not chunks:
                            result.errors.append(f"{name}: document has no text")
                            continue
                        current = store.ingest(chunks)
                        result.ingested += current.ingested
                        result.skipped += current.skipped
                        result.errors.extend(current.errors)
                        result.document_ids.extend(current.document_ids)
                    except Exception as exc:
                        logger.exception("Failed to ingest {}", name)
                        result.errors.append(f"{name}: {type(exc).__name__}")
        st.session_state.last_ingestion_result = result

    result = st.session_state.last_ingestion_result
    if result is not None:
        summary = f"{result.ingested} chunks added, {result.skipped} duplicates skipped"
        if result.errors:
            st.sidebar.warning(summary)
            for error in result.errors:
                st.sidebar.error(error)
        elif result.ingested:
            st.sidebar.success(summary)
        else:
            st.sidebar.info(summary)

    documents = store.list_documents()
    st.session_state.ingested_documents = documents
    st.sidebar.subheader("Stored documents")
    if not documents:
        st.sidebar.info("Upload a Markdown note to begin.")
    for document in documents:
        st.sidebar.write(
            f"**{document['source']}** — {document['topic']} "
            f"({document['chunk_count']} chunks)"
        )
        if st.sidebar.button("Remove", key=f"remove:{document['source']}"):
            store.delete_document(document["source"])
            st.session_state.last_ingestion_result = None
            st.rerun()
    st.sidebar.caption(
        "To replace an edited note, remove its old source before uploading it again."
    )


def render_corpus_stats(store: VectorStoreManager) -> None:
    stats = store.get_collection_stats()
    st.sidebar.metric("Total Chunks", stats["total_chunks"])
    st.sidebar.write("Topics:", ", ".join(stats["topics"]) or "None yet")
    if stats["bonus_topics_present"]:
        st.sidebar.success("Bonus topics present")


def render_document_viewer(store: VectorStoreManager) -> None:
    st.subheader("📄 Document Viewer")
    documents = st.session_state.ingested_documents
    if not documents:
        st.info("Ingest a Markdown note using the sidebar.")
        return
    source = st.selectbox(
        "Select document", options=[document["source"] for document in documents]
    )
    st.session_state.selected_document = source
    chunks = store.get_document_chunks(source)
    with st.container(height=450):
        for index, chunk in enumerate(chunks, start=1):
            st.caption(
                f"Chunk {index} | {chunk.metadata.topic} | "
                f"{chunk.metadata.difficulty} | {chunk.metadata.type}"
            )
            st.markdown(chunk.chunk_text)
            st.divider()
    st.caption(f"{len(chunks)} chunks stored for this document.")


def render_chat_interface(store: VectorStoreManager) -> None:
    """Retrieve context, call the cached LLM, and persist answers and sources."""
    st.subheader("💬 RAG Question and Answer")
    topics = store.get_collection_stats()["topics"]
    topic_column, difficulty_column = st.columns(2)
    with topic_column:
        topic = st.selectbox("Topic", ["All topics", *topics])
    with difficulty_column:
        difficulty = st.selectbox(
            "Difficulty", ["All levels", "beginner", "intermediate", "advanced"]
        )
    topic_filter = None if topic == "All topics" else topic
    difficulty_filter = None if difficulty == "All levels" else difficulty
    st.session_state.topic_filter = topic_filter
    st.session_state.difficulty_filter = difficulty_filter
    with st.container(height=450):
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    with st.expander("📎 Retrieved sources", expanded=True):
                        for source in message["sources"]:
                            st.caption(source)
                if message.get("no_context_found"):
                    st.warning("No relevant context was retrieved.")
    query = st.chat_input("Ask a question about your uploaded notes...")
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        try:
            with st.spinner("Retrieving context and generating an answer..."):
                response = answer_question(
                    query,
                    store,
                    get_llm(),
                    get_settings(),
                    topic_filter=topic_filter,
                    difficulty_filter=difficulty_filter,
                )
            message = {
                "role": "assistant",
                "content": response.answer,
                "sources": response.sources,
                "no_context_found": response.no_context_found,
            }
        except Exception as exc:
            logger.exception("RAG request failed")
            message = {
                "role": "assistant",
                "content": (
                    f"The request failed ({type(exc).__name__}). "
                    "Check your local terminal for details and your .env settings."
                ),
            }
        st.session_state.chat_history.append(message)
        st.rerun()
    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.rerun()


def main() -> None:
    settings = get_settings()
    st.set_page_config(page_title=settings.app_title, page_icon="🧠", layout="wide")
    st.title(f"🧠 {settings.app_title}")
    st.caption("Part 1 · Markdown RAG · Local embeddings · ChromaDB · Groq")
    initialise_session_state()
    try:
        store = get_vector_store()
        chunker = get_chunker()
    except Exception as exc:
        logger.exception("RAG startup failed")
        st.error(
            f"Startup failed ({type(exc).__name__}). Check the terminal for details."
        )
        st.stop()
    render_ingestion_panel(store, chunker)
    render_corpus_stats(store)
    viewer_column, chat_column = st.columns([1, 1], gap="large")
    with viewer_column:
        render_document_viewer(store)
    with chat_column:
        render_chat_interface(store)


if __name__ == "__main__":
    main()
