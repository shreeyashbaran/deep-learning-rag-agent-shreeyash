"""Part 1 integration checks; Groq responses and embeddings are test doubles."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_agent.config import LLMFactory, Settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.qa import answer_question
from rag_agent.vectorstore.store import VectorStoreManager

SAMPLE = Path(__file__).resolve().parents[1] / "examples/lstm_intermediate.md"


class FakeLLM:
    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(
            content=(
                "The forget gate controls how much of the previous cell state "
                "is retained. [SOURCE: LSTM | lstm_intermediate.md]"
            )
        )


def test_markdown_chunking_and_metadata():
    chunks = DocumentChunker().chunk_file(SAMPLE)
    assert len(chunks) > 1
    assert all(chunk.metadata.topic == "LSTM" for chunk in chunks)
    assert all(chunk.metadata.difficulty == "intermediate" for chunk in chunks)
    assert all(len(chunk.chunk_text) <= 512 for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert [chunk.chunk_id for chunk in chunks] == [
        chunk.chunk_id for chunk in DocumentChunker().chunk_file(SAMPLE)
    ]
    assert any("Forget gate" in chunk.chunk_text for chunk in chunks)


def test_metadata_override_and_bonus():
    chunker = DocumentChunker()
    metadata = chunker._infer_metadata(Path("gan_advanced.md"))
    assert metadata.topic == "GAN" and metadata.is_bonus
    metadata = chunker._infer_metadata(
        Path("test_rag.md"), {"topic": "CNN", "difficulty": "intermediate"}
    )
    assert metadata.topic == "CNN" and metadata.difficulty == "intermediate"


def test_invalid_file_and_chunk_size(tmp_path):
    chunker = DocumentChunker()
    with pytest.raises(FileNotFoundError):
        chunker.chunk_file(tmp_path / "missing.md")
    pdf = tmp_path / "test.pdf"
    pdf.write_text("Not a Markdown document")
    with pytest.raises(ValueError):
        chunker.chunk_file(pdf)
    with pytest.raises(ValueError):
        chunker.chunk_file(SAMPLE, chunk_size=50, chunk_overlap=50)


def test_empty_markdown(tmp_path):
    file = tmp_path / "empty.md"
    file.write_text(" \n")
    assert DocumentChunker().chunk_file(file) == []


def test_ingest_browse_query_delete(store):
    chunks = DocumentChunker().chunk_file(SAMPLE)
    result = store.ingest(chunks)
    assert result.ingested == len(chunks) and not result.errors
    second = store.ingest(chunks)
    assert second.ingested == 0 and second.skipped == len(chunks)
    assert store.get_collection_stats()["total_chunks"] == len(chunks)
    assert store.list_documents() == [
        {
            "source": SAMPLE.name,
            "topic": "LSTM",
            "chunk_count": len(chunks),
        }
    ]
    saved = store.get_document_chunks(SAMPLE.name)
    assert [chunk.chunk_text for chunk in saved] == [
        chunk.chunk_text for chunk in chunks
    ]
    assert store.query("LSTM forget gate", k=100)
    assert store.delete_document(SAMPLE.name) == len(chunks)
    assert store.query("LSTM forget gate") == []


def test_persistence(store, test_settings):
    chunks = DocumentChunker().chunk_file(SAMPLE)
    store.ingest(chunks)
    reopened = VectorStoreManager(test_settings)
    assert reopened.get_collection_stats()["total_chunks"] == len(chunks)
    assert reopened.query("LSTM forget gate")


def test_duplicate_within_single_batch(store):
    chunk = DocumentChunker().chunk_file(SAMPLE)[0]
    result = store.ingest([chunk, chunk])
    assert result.ingested == 1 and result.skipped == 1 and not result.errors


def test_combined_filters_and_empty_queries(store):
    store.ingest(DocumentChunker().chunk_file(SAMPLE))
    assert store.query(
        "LSTM gate", topic_filter="LSTM", difficulty_filter="intermediate"
    )
    assert store.query("LSTM gate", topic_filter="CNN") == []
    assert store.query("LSTM gate", difficulty_filter="advanced") == []
    assert store.query("") == []
    assert store.query("LSTM gate", k=0) == []


def test_batch_embedding_failure_is_reported(store, monkeypatch):
    def fail(texts):
        raise RuntimeError("test embedding failure")

    monkeypatch.setattr(store._embeddings, "embed_documents", fail)
    chunks = DocumentChunker().chunk_file(SAMPLE)
    result = store.ingest(chunks)
    assert result.ingested == 0
    assert len(result.errors) == len(chunks)
    assert store.get_collection_stats()["total_chunks"] == 0


def test_empty_chunk_is_reported(store):
    chunk = replace(DocumentChunker().chunk_file(SAMPLE)[0], chunk_text="")
    result = store.ingest([chunk])
    assert len(result.errors) == 1 and result.ingested == 0


def test_rag_prompt_contains_retrieved_text_and_sources(store, test_settings):
    store.ingest(DocumentChunker().chunk_file(SAMPLE))
    llm = FakeLLM()
    response = answer_question(
        "What does the forget gate do?", store, llm, test_settings
    )
    assert len(llm.calls) == 1
    prompt = llm.calls[0][-1].content
    assert "USER QUESTION:" in prompt and "Forget gate" in prompt
    assert "[SOURCE: LSTM | lstm_intermediate.md]" in prompt
    assert response.sources == ["[SOURCE: LSTM | lstm_intermediate.md]"]
    assert not response.no_context_found


def test_no_context_never_calls_llm(store, test_settings):
    llm = FakeLLM()
    response = answer_question("LSTM gate", store, llm, test_settings)
    assert response.no_context_found and response.sources == []
    assert llm.calls == []


def test_context_budget_limits_supplied_text(store, test_settings):
    store.ingest(DocumentChunker().chunk_file(SAMPLE))
    small = test_settings.model_copy(update={"max_context_tokens": 35})
    llm = FakeLLM()
    answer_question("LSTM forget gate", store, llm, small)
    import tiktoken

    context = llm.calls[0][-1].content.split("RETRIEVED CONTEXT:\n", 1)[1]
    context = context.split("\n\nUSER QUESTION:", 1)[0]
    assert len(tiktoken.get_encoding("cl100k_base").encode(context)) <= 35


def test_groq_factory_rejects_missing_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    settings = Settings(_env_file=None, GROQ_API_KEY="")
    with pytest.raises(EnvironmentError, match="GROQ_API_KEY is missing"):
        LLMFactory(settings).create()


def test_streamlit_startup_chat_and_history(store, test_settings, monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    import rag_agent.config as config
    import rag_agent.corpus.chunker as chunker_module
    import rag_agent.vectorstore.store as store_module

    store.ingest(DocumentChunker().chunk_file(SAMPLE))
    monkeypatch.setattr(config, "get_settings", lambda: test_settings)
    monkeypatch.setattr(chunker_module, "get_settings", lambda: test_settings)
    monkeypatch.setattr(store_module, "get_settings", lambda: test_settings)
    monkeypatch.setattr(LLMFactory, "create", lambda self: FakeLLM())
    st.cache_resource.clear()
    app = Path(__file__).resolve().parents[1] / "src/rag_agent/ui/app.py"
    ui = AppTest.from_file(str(app), default_timeout=30).run()
    assert not ui.exception
    assert int(ui.metric[0].value) > 0
    ui.chat_input[0].set_value("What does the forget gate do?").run()
    assert not ui.exception
    assert len(ui.chat_message) == 2
    assert "previous cell state" in ui.chat_message[1].markdown[0].value
    assert any("lstm_intermediate.md" in item.value for item in ui.caption)
    ui.run()
    assert len(ui.chat_message) == 2
    st.cache_resource.clear()
