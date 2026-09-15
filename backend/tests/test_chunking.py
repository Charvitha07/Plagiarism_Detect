import pytest

from app.services.chunking import chunk_sentences
from app.services.sentence_split import RegexSentenceSplitter
from tests.conftest import make_sentence


def _sentences(word_counts, **kwargs):
    """Builds a run of sentences with the given word counts, auto-numbered.
    Each sentence starts with a unique capitalized token ("Sentence{i}") so
    they're never textually identical to one another (tests need to tell
    them apart) and so RegexSentenceSplitter — which looks for a capital
    letter after the terminal period — can correctly split them back
    apart when a test round-trips a chunk's joined text."""
    out = []
    for i, n in enumerate(word_counts, start=1):
        words = [f"Sentence{i}", *(f"w{j}" for j in range(n - 1))]
        text = " ".join(words) + "."
        out.append(make_sentence(text, sentence_number=i, **kwargs))
    return out


def test_empty_input_returns_no_chunks():
    assert chunk_sentences([]) == []


def test_invalid_word_bounds_raise():
    with pytest.raises(ValueError):
        chunk_sentences(_sentences([10]), min_words=50, max_words=10)


def test_negative_overlap_raises():
    with pytest.raises(ValueError):
        chunk_sentences(_sentences([10]), overlap_sentences=-1)


def test_chunks_respect_max_words_when_possible():
    sentences = _sentences([15, 15, 15, 15, 15])  # 5 sentences x 15 words = 75
    chunks = chunk_sentences(sentences, min_words=20, max_words=40, overlap_sentences=0)
    assert all(c.word_count <= 40 for c in chunks)
    # No sentence is lost or duplicated across chunks when there's no overlap.
    assert sum(c.sentence_count for c in chunks) == len(sentences)


def test_chunk_never_splits_a_sentence():
    sentences = _sentences([15, 15, 15])
    chunks = chunk_sentences(sentences, min_words=10, max_words=20)
    for chunk in chunks:
        # word_count on the chunk must equal the words actually in `text`.
        assert chunk.word_count == len(chunk.text.split())


def test_oversized_single_sentence_becomes_its_own_chunk():
    sentences = _sentences([5, 200, 5])
    chunks = chunk_sentences(sentences, min_words=20, max_words=80)
    assert any(c.sentence_count == 1 and c.word_count == 200 for c in chunks)


def test_chunk_ids_are_unique():
    sentences = _sentences([10, 10, 10, 10])
    chunks = chunk_sentences(sentences, min_words=5, max_words=15)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_text_preserves_sentence_order():
    sentences = [
        make_sentence("Alpha sentence.", sentence_number=1),
        make_sentence("Beta sentence.", sentence_number=2),
    ]
    chunks = chunk_sentences(sentences, min_words=1, max_words=100)
    assert chunks[0].text == "Alpha sentence. Beta sentence."


def test_provenance_spans_first_to_last_sentence():
    sentences = [
        make_sentence("One two three four five.", paragraph_number=1, sentence_number=1, page_number=1),
        make_sentence("Six seven eight nine ten.", paragraph_number=2, sentence_number=2, page_number=1),
        make_sentence("Eleven twelve thirteen.", paragraph_number=3, sentence_number=3, page_number=2),
    ]
    chunks = chunk_sentences(sentences, min_words=1, max_words=100)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.paragraph_start == 1 and chunk.paragraph_end == 3
    assert chunk.sentence_start == 1 and chunk.sentence_end == 3
    assert chunk.page_start == 1 and chunk.page_end == 2


def test_no_overlap_means_disjoint_chunks():
    sentences = _sentences([20, 20, 20, 20])
    chunks = chunk_sentences(sentences, min_words=15, max_words=25, overlap_sentences=0)
    splitter = RegexSentenceSplitter()
    seen_sentence_texts = []
    for chunk in chunks:
        seen_sentence_texts.extend(splitter.split(chunk.text))
    # Every original sentence appears exactly once across all chunks.
    assert sorted(seen_sentence_texts) == sorted(s.text for s in sentences)


def test_overlap_carries_the_boundary_sentence_into_the_next_chunk():
    sentences = _sentences([20, 20, 20, 20])
    chunks = chunk_sentences(sentences, min_words=15, max_words=25, overlap_sentences=1)
    assert len(chunks) >= 2

    splitter = RegexSentenceSplitter()
    for prev_chunk, next_chunk in zip(chunks, chunks[1:]):
        prev_last_sentence = splitter.split(prev_chunk.text)[-1]
        next_first_sentence = splitter.split(next_chunk.text)[0]
        assert next_first_sentence == prev_last_sentence
