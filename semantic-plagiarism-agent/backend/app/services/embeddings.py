"""
Turns text into normalized semantic embeddings, batched.

`EmbeddingBackend` is a small interface so the rest of the app (chunking
output -> vectors -> FAISS in Phase 2) never talks to sentence-transformers
directly. Two implementations:

- `SentenceTransformerBackend` — the real thing. Imports `sentence_transformers`
  lazily so importing this module (and running the rest of the test suite)
  doesn't require torch/sentence-transformers to be installed, and doesn't
  require network access to download model weights. Configure the model
  via `Settings.embedding_model_name` (default
  `sentence-transformers/all-MiniLM-L6-v2`; also supports
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`).
- `DeterministicHashBackend` — a fast, dependency-free stand-in used in
  tests and local dev without internet access. NOT semantically
  meaningful; it exists purely to exercise batching/normalization logic
  and the rest of the pipeline deterministically.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import List, Sequence

import numpy as np

from app.core.errors import EmbeddingBackendUnavailableError
from app.schemas.document import DocumentChunk


class EmbeddingBackend(ABC):
    """Minimal interface every embedding backend must satisfy."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Returns a (len(texts), dimension) float32 array. Not required
        to be normalized — `EmbeddingService` normalizes centrally."""
        ...


class SentenceTransformerBackend(EmbeddingBackend):
    """Real backend, backed by `sentence-transformers`.

    Requires `pip install sentence-transformers torch` and network access
    (or a local cache) to fetch the model weights the first time it's
    used. Both are expected to be available in the actual deployment
    environment; they are intentionally NOT required just to import this
    module or run the unit test suite.
    """

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingBackendUnavailableError(
                "sentence-transformers is not installed. Run: "
                "pip install sentence-transformers torch"
            ) from exc
        try:
            self._model = SentenceTransformer(model_name, device=device)
        except Exception as exc:  # model not found / no network / corrupt cache
            raise EmbeddingBackendUnavailableError(
                f"Could not load embedding model '{model_name}': {exc}"
            ) from exc
        self.model_name = model_name

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts), convert_to_numpy=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype=np.float32)


class DeterministicHashBackend(EmbeddingBackend):
    """Dependency-free fake embedder for tests / offline dev.

    Produces a deterministic pseudo-random vector per input string (same
    text -> same vector, always), so pipeline logic — batching, shapes,
    normalization, downstream similarity math — can be fully exercised
    without downloading any model. It captures no real semantics.
    """

    def __init__(self, dimension: int = 32) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        rows = [self._vector_for(t) for t in texts]
        return np.vstack(rows) if rows else np.zeros((0, self._dimension), dtype=np.float32)

    def _vector_for(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big", signed=False)
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self._dimension).astype(np.float32)


def normalize_embeddings(matrix: np.ndarray) -> np.ndarray:
    """L2-normalizes each row; zero vectors are left as zero (no divide-by-zero)."""
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class EmbeddingService:
    """Batches text through an `EmbeddingBackend` and L2-normalizes the result."""

    def __init__(self, backend: EmbeddingBackend, batch_size: int = 32) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.backend = backend
        self.batch_size = batch_size

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.backend.dimension), dtype=np.float32)

        batches: List[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            batches.append(self.backend.encode(batch))
        matrix = np.vstack(batches).astype(np.float32)
        return normalize_embeddings(matrix)

    def embed_chunks(self, chunks: Sequence[DocumentChunk]) -> np.ndarray:
        return self.embed_texts([c.text for c in chunks])
