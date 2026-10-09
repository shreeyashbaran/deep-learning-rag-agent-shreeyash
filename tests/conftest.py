"""Use real Chroma storage with deterministic test embeddings; no API keys."""

from __future__ import annotations

import re

import pytest

from rag_agent.config import EmbeddingFactory, Settings
from rag_agent.vectorstore.store import VectorStoreManager


class TestEmbeddings:
    __test__ = False

    def embed_query(self, text):
        words = set(re.findall(r"[a-z]+", text.lower()))
        groups = [
            {"lstm", "gate", "gates", "forget", "recurrent", "cell"},
            {"gan", "generative", "adversarial", "generator", "discriminator"},
            {"roman", "empire", "history"},
            {"cnn", "convolution", "pooling"},
        ]
        return [float(len(words & group)) for group in groups] + [0.01]

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]


@pytest.fixture
def test_settings(tmp_path):
    return Settings(
        _env_file=None,
        CHROMA_DB_PATH=str(tmp_path / "chroma"),
        CHROMA_COLLECTION_NAME="workshop_test",
        SIMILARITY_THRESHOLD=0.3,
        RETRIEVAL_K=4,
    )


@pytest.fixture
def store(monkeypatch, test_settings):
    monkeypatch.setattr(EmbeddingFactory, "create", lambda self: TestEmbeddings())
    return VectorStoreManager(test_settings)
