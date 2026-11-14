"""Local sentence-transformers embedding wrapper. No API calls, no cost."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_DEFAULT_MODEL = "all-MiniLM-L6-v2"  


@dataclass(frozen=True)
class EmbedderConfig:
    model_name: str = _DEFAULT_MODEL
    batch_size: int = 32
    normalize: bool = True


class Embedder:
    def __init__(self, config: EmbedderConfig | None = None) -> None:
        self.config = config or EmbedderConfig()
        self._model = None

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.config.model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        """texts: list of strings -> (n, dim) float32 array."""
        self._load()
        embeddings = self._model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=len(texts) > 50,
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]