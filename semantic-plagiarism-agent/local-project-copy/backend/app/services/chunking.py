"""
Groups consecutive `ExtractedSentence` objects into `DocumentChunk`s for
embedding, without ever splitting a sentence across two chunks.

Strategy: greedily add sentences to the current chunk; close the chunk
once it has at least `min_words` and adding the next sentence would push
it past `max_words`. A single sentence longer than `max_words` still
becomes its own (oversized) chunk rather than being cut — chunk
boundaries always fall on sentence boundaries.

`overlap_sentences` carries the trailing N sentences of a closed chunk
into the start of the next one, which helps retrieval catch paraphrases
that straddle a chunk boundary.
"""
from __future__ import annotations

from typing import List
from uuid import uuid4

from app.schemas.document import DocumentChunk, ExtractedSentence


def chunk_sentences(
    sentences: List[ExtractedSentence],
    min_words: int = 20,
    max_words: int = 80,
    overlap_sentences: int = 0,
) -> List[DocumentChunk]:
    if not sentences:
        return []
    if min_words < 1 or max_words < 1 or min_words > max_words:
        raise ValueError("Require 1 <= min_words <= max_words")
    if overlap_sentences < 0:
        raise ValueError("overlap_sentences must be >= 0")

    chunks: List[DocumentChunk] = []
    current: List[ExtractedSentence] = []
    current_words = 0

    def close_chunk() -> None:
        nonlocal current, current_words
        if not current:
            return
        chunks.append(_build_chunk(current))
        overlap = current[-overlap_sentences:] if overlap_sentences else []
        current = list(overlap)
        current_words = sum(s.word_count for s in current)

    for sentence in sentences:
        projected = current_words + sentence.word_count
        would_overflow = bool(current) and projected > max_words
        # Close the in-progress chunk before adding this sentence if either:
        # (a) the chunk already meets min_words and adding would overflow it, or
        # (b) this sentence is itself oversized and would overflow no matter
        #     what — in that case an already-open small chunk shouldn't
        #     absorb it; let the oversized sentence start its own chunk.
        sentence_is_oversized = sentence.word_count >= max_words
        if would_overflow and (current_words >= min_words or sentence_is_oversized):
            close_chunk()
        current.append(sentence)
        current_words += sentence.word_count
        if current_words >= max_words:
            close_chunk()

    close_chunk()
    return chunks


def _build_chunk(sentences: List[ExtractedSentence]) -> DocumentChunk:
    text = " ".join(s.text for s in sentences)
    first, last = sentences[0], sentences[-1]
    return DocumentChunk(
        chunk_id=str(uuid4()),
        filename=first.filename,
        page_start=first.page_number,
        page_end=last.page_number,
        paragraph_start=first.paragraph_number,
        paragraph_end=last.paragraph_number,
        sentence_start=first.sentence_number,
        sentence_end=last.sentence_number,
        text=text,
        word_count=sum(s.word_count for s in sentences),
        sentence_count=len(sentences),
    )
