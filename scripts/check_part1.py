"""Exercise the actual local embedding model and optionally the Groq API."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from rag_agent.config import LLMFactory, get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.qa import answer_question
from rag_agent.vectorstore.store import VectorStoreManager


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--groq", action="store_true", help="Also make one Groq request"
    )
    args = parser.parse_args()
    sample = Path(__file__).resolve().parents[1] / "examples/lstm_intermediate.md"
    with TemporaryDirectory(prefix="rag-check-", ignore_cleanup_errors=True) as folder:
        settings = get_settings().model_copy(update={"chroma_db_path": folder})
        store = VectorStoreManager(settings)
        chunks = DocumentChunker(settings).chunk_file(sample)
        first = store.ingest(chunks)
        assert first.ingested == len(chunks) > 0 and not first.errors, first.errors
        second = store.ingest(chunks)
        assert second.skipped == len(chunks) and second.ingested == 0
        query = "What does the LSTM forget gate do?"
        retrieved = store.query(query)
        assert retrieved, "No relevant chunks; inspect the similarity threshold."
        assert any("forget gate" in chunk.chunk_text.lower() for chunk in retrieved)
        print(f"PASS: {len(chunks)} chunks ingested; duplicate upload skipped.")
        print(f"PASS: retrieved {len(retrieved)} relevant chunks.")
        for chunk in retrieved:
            print(f"  {chunk.to_citation()} similarity={chunk.score:.3f}")
        if args.groq:
            llm = LLMFactory(settings).create()
            response = answer_question(query, store, llm, settings)
            assert response.answer.strip() and response.sources
            print("PASS: Groq returned an answer with retrieved sources.")
            print(response.answer)
            print("Sources:", ", ".join(response.sources))


if __name__ == "__main__":
    main()
