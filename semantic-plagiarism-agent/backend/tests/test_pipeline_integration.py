"""Exercises the full Phase 1 pipeline end to end: file -> sentences ->
chunks -> normalized embeddings, across all three supported file types."""
import numpy as np
import pytest

from app.services.chunking import chunk_sentences
from app.services.embeddings import DeterministicHashBackend, EmbeddingService
from app.services.extraction import extract_document


@pytest.mark.parametrize("fixture_name", ["sample_txt", "sample_docx", "sample_pdf"])
def test_full_pipeline_produces_normalized_embeddings(fixture_name, request):
    path = request.getfixturevalue(fixture_name)

    sentences = extract_document(path)
    assert len(sentences) > 0

    chunks = chunk_sentences(sentences, min_words=5, max_words=60)
    assert len(chunks) > 0
    assert sum(c.sentence_count for c in chunks) == len(sentences)

    service = EmbeddingService(DeterministicHashBackend(dimension=16), batch_size=4)
    vectors = service.embed_chunks(chunks)

    assert vectors.shape == (len(chunks), 16)
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_pipeline_is_deterministic_across_runs(sample_txt):
    def run():
        sentences = extract_document(sample_txt)
        chunks = chunk_sentences(sentences, min_words=5, max_words=60)
        service = EmbeddingService(DeterministicHashBackend(dimension=16))
        return service.embed_chunks(chunks)

    vectors_a = run()
    vectors_b = run()
    assert np.allclose(vectors_a, vectors_b)
