"""
store.py
========
ChromaDB vector store management.

Handles all interactions with the persistent ChromaDB collection:
initialisation, ingestion, duplicate detection, and retrieval.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from loguru import logger

from rag_agent.agent.state import (
    ChunkMetadata,
    DocumentChunk,
    IngestionResult,
    RetrievedChunk,
)
from rag_agent.config import EmbeddingFactory, Settings, get_settings


class VectorStoreManager:
    """
    Manages the ChromaDB persistent vector store for the corpus.

    All corpus ingestion and retrieval operations pass through this class.
    It is the single point of contact between the application and ChromaDB.

    Parameters
    ----------
    settings : Settings, optional
        Application settings. Uses get_settings() singleton if not provided.

    Example
    -------
    >>> manager = VectorStoreManager()
    >>> result = manager.ingest(chunks)
    >>> print(f"Ingested: {result.ingested}, Skipped: {result.skipped}")
    >>>
    >>> chunks = manager.query("explain the vanishing gradient problem", k=4)
    >>> for chunk in chunks:
    ...     print(chunk.to_citation(), chunk.score)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._embeddings = EmbeddingFactory(self._settings).create()
        self._client = None
        self._collection = None
        self._initialise()

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def _initialise(self) -> None:
        """
        Create or connect to the persistent ChromaDB client and collection.

        Creates the chroma_db_path directory if it does not exist.
        Uses PersistentClient so data survives between application restarts.

        Called automatically during __init__. Should not be called directly.

        Raises
        ------
        RuntimeError
            If ChromaDB cannot be initialised at the configured path.
        """
        try:
            import chromadb

            db_path = Path(self._settings.chroma_db_path)
            db_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(db_path))
            self._collection = self._client.get_or_create_collection(
                name=self._settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=None,
            )
            logger.info(
                "Initialised ChromaDB collection '{}' with {} items",
                self._settings.chroma_collection_name,
                self._collection.count(),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialise ChromaDB at '{self._settings.chroma_db_path}'"
            ) from exc

    # -----------------------------------------------------------------------
    # Duplicate Detection
    # -----------------------------------------------------------------------

    @staticmethod
    def generate_chunk_id(source: str, chunk_text: str) -> str:
        """
        Generate a deterministic chunk ID from source filename and content.

        Using a content hash ensures two uploads of the same file produce
        the same IDs, making duplicate detection reliable for repeated uploads
        with the same source filename and content.

        Parameters
        ----------
        source : str
            The source filename (e.g. 'lstm.md').
        chunk_text : str
            The full text content of the chunk.

        Returns
        -------
        str
            A 16-character hex string derived from SHA-256 of the inputs.
        """
        content = f"{source}::{chunk_text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def check_duplicate(self, chunk_id: str) -> bool:
        """
        Check whether a chunk with this ID already exists in the collection.

        Parameters
        ----------
        chunk_id : str
            The deterministic chunk ID to check.

        Returns
        -------
        bool
            True if the chunk already exists (duplicate). False otherwise.

        Interview talking point: content-addressed deduplication is more
        robust than filename-based deduplication because it detects repeated
        uploads with the same source filename and content.
        """
        result = self._collection.get(ids=[chunk_id], include=[])
        return chunk_id in result["ids"]

    # -----------------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------------

    def ingest(self, chunks: list[DocumentChunk]) -> IngestionResult:
        """
        Embed and store a list of DocumentChunks in ChromaDB.

        Checks each chunk for duplicates before embedding. Skips duplicates
        silently and records the count in the returned IngestionResult.

        Parameters
        ----------
        chunks : list[DocumentChunk]
            Prepared chunks with text and metadata. Use DocumentChunker
            to produce these from raw files.

        Returns
        -------
        IngestionResult
            Summary with counts of ingested, skipped, and errored chunks.

        Notes
        -----
        Embeds in batches of 100 to avoid memory issues with large corpora.
        Uses upsert (not add) so insertion is idempotent for each chunk ID.
        Modified text produces a new ID; remove the old document before
        ingesting a replacement.

        Interview talking point: batch processing with a configurable
        batch size is a production pattern that prevents OOM errors when
        ingesting large document sets.
        """
        result = IngestionResult()
        pending = []
        seen = set()
        for chunk in chunks:
            try:
                if not chunk.chunk_text.strip():
                    raise ValueError("Chunk text is empty")
                if chunk.chunk_id in seen or self.check_duplicate(chunk.chunk_id):
                    result.skipped += 1
                    continue
                seen.add(chunk.chunk_id)
                pending.append(chunk)
            except Exception as exc:
                result.errors.append(f"{chunk.chunk_id}: {exc}")

        for offset in range(0, len(pending), 100):
            batch = pending[offset : offset + 100]
            try:
                vectors = self._embeddings.embed_documents(
                    [chunk.chunk_text for chunk in batch]
                )
                if len(vectors) != len(batch):
                    raise ValueError("Embedding count does not match chunk count")
                metadatas = []
                for chunk in batch:
                    metadata = chunk.metadata.to_dict()
                    metadata["chunk_index"] = chunk.chunk_index
                    metadatas.append(metadata)
                self._collection.upsert(
                    ids=[chunk.chunk_id for chunk in batch],
                    embeddings=vectors,
                    documents=[chunk.chunk_text for chunk in batch],
                    metadatas=metadatas,
                )
                result.ingested += len(batch)
                for chunk in batch:
                    if chunk.metadata.source not in result.document_ids:
                        result.document_ids.append(chunk.metadata.source)
            except Exception as exc:
                for chunk in batch:
                    result.errors.append(f"{chunk.chunk_id}: {exc}")
        logger.info(
            "Ingestion: {} added, {} skipped, {} errors",
            result.ingested,
            result.skipped,
            len(result.errors),
        )
        return result

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        k: int | None = None,
        topic_filter: str | None = None,
        difficulty_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Applies similarity threshold filtering — chunks below
        settings.similarity_threshold are excluded from results.

        Parameters
        ----------
        query_text : str
            The user query or rewritten query to retrieve against.
        k : int, optional
            Number of chunks to retrieve. Defaults to settings.retrieval_k.
        topic_filter : str, optional
            Restrict retrieval to a specific topic (e.g. 'LSTM').
            Maps to ChromaDB where-filter on metadata.topic.
        difficulty_filter : str, optional
            Restrict retrieval to a difficulty level.
            Maps to ChromaDB where-filter on metadata.difficulty.

        Returns
        -------
        list[RetrievedChunk]
            Chunks sorted by similarity score descending.
            Empty list if no chunks meet the similarity threshold.

        Interview talking point: returning an empty list (not hallucinating)
        when no relevant context exists is the hallucination guard. This is
        a critical production RAG pattern — the system must know what it
        does not know.
        """
        limit = self._settings.retrieval_k if k is None else k
        count = self._collection.count()
        if not query_text.strip() or limit <= 0 or count == 0:
            return []
        filters = []
        if topic_filter:
            filters.append({"topic": topic_filter})
        if difficulty_filter:
            filters.append({"difficulty": difficulty_filter})
        where_filter = None
        if len(filters) == 1:
            where_filter = filters[0]
        elif len(filters) > 1:
            where_filter = {"$and": filters}
        response = self._collection.query(
            query_embeddings=[self._embeddings.embed_query(query_text)],
            n_results=min(limit, count),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
        retrieved = []
        for chunk_id, text, metadata, distance in zip(
            response["ids"][0],
            response["documents"][0],
            response["metadatas"][0],
            response["distances"][0],
        ):
            score = 1.0 - float(distance)
            if score >= self._settings.similarity_threshold:
                retrieved.append(
                    RetrievedChunk(
                        chunk_id=chunk_id,
                        chunk_text=text,
                        metadata=ChunkMetadata.from_dict(metadata),
                        score=score,
                    )
                )
        return sorted(retrieved, key=lambda chunk: chunk.score, reverse=True)

    # -----------------------------------------------------------------------
    # Corpus Inspection
    # -----------------------------------------------------------------------

    def list_documents(self) -> list[dict]:
        """
        Return a list of all unique source documents in the collection.

        Used by the UI to populate the document viewer panel.

        Returns
        -------
        list[dict]
            Each item contains: source (str), topic (str), chunk_count (int).
        """
        response = self._collection.get(include=["metadatas"])
        documents = {}
        for metadata in response["metadatas"] or []:
            source = metadata["source"]
            if source not in documents:
                documents[source] = {
                    "source": source,
                    "topic": metadata["topic"],
                    "chunk_count": 0,
                }
            documents[source]["chunk_count"] += 1
        return [documents[source] for source in sorted(documents)]

    def get_document_chunks(self, source: str) -> list[DocumentChunk]:
        """
        Retrieve all chunks belonging to a specific source document.

        Used by the document viewer to display document content.

        Parameters
        ----------
        source : str
            The source filename to retrieve chunks for.

        Returns
        -------
        list[DocumentChunk]
            All chunks from this source, ordered by their position
            in the original document.
        """
        response = self._collection.get(
            where={"source": source}, include=["documents", "metadatas"]
        )
        chunks = [
            DocumentChunk(
                chunk_id=chunk_id,
                chunk_text=text,
                metadata=ChunkMetadata.from_dict(metadata),
                chunk_index=int(metadata.get("chunk_index", 0)),
            )
            for chunk_id, text, metadata in zip(
                response["ids"], response["documents"], response["metadatas"]
            )
        ]
        return sorted(chunks, key=lambda chunk: chunk.chunk_index)

    def get_collection_stats(self) -> dict:
        """
        Return summary statistics about the current collection.

        Used by the UI to show corpus health at a glance.

        Returns
        -------
        dict
            Keys: total_chunks, topics (list), sources (list),
            bonus_topics_present (bool).
        """
        metadata = self._collection.get(include=["metadatas"])["metadatas"] or []
        return {
            "total_chunks": self._collection.count(),
            "topics": sorted({item["topic"] for item in metadata}),
            "sources": sorted({item["source"] for item in metadata}),
            "bonus_topics_present": any(
                str(item.get("is_bonus", "false")).lower() == "true"
                for item in metadata
            ),
        }

    def delete_document(self, source: str) -> int:
        """
        Remove all chunks from a specific source document.

        Parameters
        ----------
        source : str
            Source filename to remove.

        Returns
        -------
        int
            Number of chunks deleted.
        """
        ids = self._collection.get(where={"source": source}, include=[])["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)
