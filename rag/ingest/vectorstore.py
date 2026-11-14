"""ChromaDB vector store wrapper with metadata filtering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ingest.chunker import Chunk

_DEFAULT_PERSIST_DIR = Path(__file__).resolve().parents[2] / "rag" / "db"
_COLLECTION_NAME = "crop_disease_advisor"


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    source_id: str
    title: str
    section: str
    text: str
    publisher: str
    distance: float


class VectorStore:
    def __init__(self, persist_dir: Path | None = None) -> None:
        import chromadb
        
        self._persist_dir = persist_dir or _DEFAULT_PERSIST_DIR
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def __enter__(self) -> VectorStore:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """Stops the underlying Chroma client system and releases file locks."""
        if hasattr(self, "_collection"):
            self._collection = None
        if hasattr(self, "_client") and self._client:
            if hasattr(self._client, "_system") and self._client._system:
                try:
                    self._client._system.stop()
                except Exception:
                    pass
            self._client = None
        import gc
        gc.collect()

    def add_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if not chunks:
            return
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings.tolist(),
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "source_id": c.source_id,
                    "title": c.title,
                    "section": c.section,
                    "publisher": c.publisher,
                    "crops": ",".join(c.crops),
                    "diseases": ",".join(c.diseases),
                    "regions": ",".join(c.regions),
                }
                for c in chunks
            ],
        )

    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5,
        crop_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieves top-n chunks. Optionally filters by crop."""
        where: dict | None = None
        if crop_filter and crop_filter != "all":
            where = {"$or": [
                {"crops": {"$contains": crop_filter}},
                {"crops": {"$eq": "all"}},
            ]}

        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        for i, chunk_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            retrieved.append(RetrievedChunk(
                chunk_id=chunk_id,
                source_id=meta["source_id"],
                title=meta["title"],
                section=meta["section"],
                text=results["documents"][0][i],
                publisher=meta["publisher"],
                distance=results["distances"][0][i],
            ))
        return retrieved

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        """Drops and recreates the collection. Used during re-ingestion."""
        import chromadb

        self._client.delete_collection(_COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )