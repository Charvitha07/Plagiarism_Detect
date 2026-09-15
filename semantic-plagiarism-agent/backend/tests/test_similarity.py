import numpy as np
import pytest

from app.services.similarity import cosine_similarity_matrix, top_k_matches


def test_identical_vectors_have_similarity_one():
    v = np.array([[1.0, 2.0, 3.0]])
    sims = cosine_similarity_matrix(v, v)
    assert np.isclose(sims[0, 0], 1.0, atol=1e-6)


def test_orthogonal_vectors_have_similarity_zero():
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.0, 1.0]])
    sims = cosine_similarity_matrix(a, b)
    assert np.isclose(sims[0, 0], 0.0, atol=1e-6)


def test_opposite_vectors_have_similarity_negative_one():
    a = np.array([[1.0, 0.0]])
    b = np.array([[-1.0, 0.0]])
    sims = cosine_similarity_matrix(a, b)
    assert np.isclose(sims[0, 0], -1.0, atol=1e-6)


def test_similarity_is_scale_invariant():
    a = np.array([[2.0, 0.0]])
    b = np.array([[10.0, 0.0]])
    sims = cosine_similarity_matrix(a, b)
    assert np.isclose(sims[0, 0], 1.0, atol=1e-6)


def test_matrix_shape_matches_inputs():
    a = np.random.default_rng(0).standard_normal((3, 5))
    b = np.random.default_rng(1).standard_normal((7, 5))
    sims = cosine_similarity_matrix(a, b)
    assert sims.shape == (3, 7)


def test_empty_inputs_return_empty_matrix():
    a = np.zeros((0, 4))
    b = np.random.default_rng(0).standard_normal((3, 4))
    sims = cosine_similarity_matrix(a, b)
    assert sims.shape == (0, 3)


def test_top_k_matches_orders_by_similarity_descending():
    query = np.array([1.0, 0.0])
    corpus = np.array(
        [
            [0.0, 1.0],   # sim 0
            [1.0, 0.0],   # sim 1 (identical)
            [0.7, 0.7],   # sim ~0.707
        ]
    )
    results = top_k_matches(query, corpus, k=3)
    indices_in_order = [idx for idx, _ in results]
    assert indices_in_order == [1, 2, 0]
    assert results[0][1] > results[1][1] > results[2][1]


def test_top_k_matches_respects_k():
    query = np.array([1.0, 0.0])
    corpus = np.random.default_rng(0).standard_normal((10, 2))
    results = top_k_matches(query, corpus, k=3)
    assert len(results) == 3


def test_top_k_matches_k_larger_than_corpus_is_clamped():
    query = np.array([1.0, 0.0])
    corpus = np.array([[1.0, 0.0], [0.0, 1.0]])
    results = top_k_matches(query, corpus, k=10)
    assert len(results) == 2


def test_top_k_matches_empty_corpus_returns_empty_list():
    query = np.array([1.0, 0.0])
    corpus = np.zeros((0, 2))
    assert top_k_matches(query, corpus, k=5) == []


def test_top_k_matches_invalid_k_raises():
    query = np.array([1.0, 0.0])
    corpus = np.array([[1.0, 0.0]])
    with pytest.raises(ValueError):
        top_k_matches(query, corpus, k=0)
