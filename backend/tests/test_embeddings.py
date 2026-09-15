import numpy as np
import pytest

from app.core.errors import EmbeddingBackendUnavailableError
from app.services.embeddings import (
    DeterministicHashBackend,
    EmbeddingService,
    SentenceTransformerBackend,
    normalize_embeddings,
)


class CountingBackend(DeterministicHashBackend):
    """Wraps DeterministicHashBackend but records how it was called, so
    batching behavior can be asserted without needing a real ML model."""

    def __init__(self, dimension: int = 16, batch_sizes_seen: list | None = None):
        super().__init__(dimension=dimension)
        self.batch_sizes_seen = batch_sizes_seen if batch_sizes_seen is not None else []

    def encode(self, texts):
        self.batch_sizes_seen.append(len(texts))
        return super().encode(texts)


def test_normalize_embeddings_produces_unit_vectors():
    matrix = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    normalized = normalize_embeddings(matrix)
    norms = np.linalg.norm(normalized, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_normalize_embeddings_handles_zero_vector():
    matrix = np.array([[0.0, 0.0]], dtype=np.float32)
    normalized = normalize_embeddings(matrix)
    assert np.allclose(normalized, 0.0)  # no divide-by-zero / NaN


def test_normalize_embeddings_handles_empty_matrix():
    matrix = np.zeros((0, 4), dtype=np.float32)
    assert normalize_embeddings(matrix).shape == (0, 4)


def test_embed_texts_returns_correct_shape():
    service = EmbeddingService(DeterministicHashBackend(dimension=16), batch_size=32)
    vectors = service.embed_texts(["hello world", "another sentence", "a third one"])
    assert vectors.shape == (3, 16)


def test_embed_texts_output_is_normalized():
    service = EmbeddingService(DeterministicHashBackend(dimension=16), batch_size=32)
    vectors = service.embed_texts(["some text here"])
    assert np.isclose(np.linalg.norm(vectors[0]), 1.0, atol=1e-5)


def test_embed_texts_empty_list_returns_empty_array():
    service = EmbeddingService(DeterministicHashBackend(dimension=16))
    vectors = service.embed_texts([])
    assert vectors.shape == (0, 16)


def test_same_text_always_produces_same_embedding():
    service = EmbeddingService(DeterministicHashBackend(dimension=16))
    v1 = service.embed_texts(["repeated text"])[0]
    v2 = service.embed_texts(["repeated text"])[0]
    assert np.allclose(v1, v2)


def test_different_text_produces_different_embeddings():
    service = EmbeddingService(DeterministicHashBackend(dimension=16))
    v1 = service.embed_texts(["completely different content one"])[0]
    v2 = service.embed_texts(["something else entirely two"])[0]
    assert not np.allclose(v1, v2)


def test_batching_respects_batch_size():
    backend = CountingBackend(dimension=8)
    service = EmbeddingService(backend, batch_size=2)
    texts = [f"sentence number {i}" for i in range(5)]  # 5 texts, batch_size=2 -> 3 calls
    service.embed_texts(texts)
    assert backend.batch_sizes_seen == [2, 2, 1]


def test_embed_chunks_uses_chunk_text(tmp_path):
    from app.services.chunking import chunk_sentences
    from tests.conftest import make_sentence

    sentences = [make_sentence("Some words here for a chunk.", sentence_number=1)]
    chunks = chunk_sentences(sentences, min_words=1, max_words=50)

    service = EmbeddingService(DeterministicHashBackend(dimension=8))
    vectors = service.embed_chunks(chunks)
    assert vectors.shape == (1, 8)


def test_invalid_batch_size_raises():
    with pytest.raises(ValueError):
        EmbeddingService(DeterministicHashBackend(dimension=8), batch_size=0)


def test_sentence_transformer_backend_raises_clear_error_when_dependency_missing():
    """Phase 1 deliberately doesn't require sentence-transformers/torch to
    be installed to run the test suite. When they're absent, constructing
    the real backend should fail with a clear, actionable
    EmbeddingBackendUnavailableError — not a raw ImportError leaking out
    of an unrelated module."""
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("sentence-transformers is installed here; import-error path isn't exercised")

    with pytest.raises(EmbeddingBackendUnavailableError):
        SentenceTransformerBackend("sentence-transformers/all-MiniLM-L6-v2")
