"""
Core dataclasses for the extraction -> chunking -> embedding pipeline.

These are deliberately plain `dataclasses`, not Pydantic models or
SQLAlchemy models: they're the in-memory representation used *inside*
the NLP pipeline. Phase 3 will map `DocumentChunk` rows into the
`DocumentChunk` SQLAlchemy model and Pydantic API schemas; keeping this
layer framework-free keeps the pipeline testable and reusable (e.g. from
a CLI or a notebook) without pulling in FastAPI/SQLAlchemy.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SupportedFileType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"

    @classmethod
    def from_suffix(cls, suffix: str) -> "SupportedFileType":
        cleaned = suffix.lower().lstrip(".")
        for member in cls:
            if member.value == cleaned:
                return member
        raise ValueError(f"Unsupported file suffix: {suffix!r}")


@dataclass(frozen=True, slots=True)
class ExtractedSentence:
    """One sentence pulled out of a source document, with full provenance.

    `page_number` is 1-based and only present for paginated formats (PDF).
    `paragraph_number` and `sentence_number` are 1-based and monotonically
    increasing across the *whole* document (not reset per page), which
    keeps chunk boundaries unambiguous.
    """

    filename: str
    page_number: Optional[int]
    paragraph_number: int
    sentence_number: int
    text: str

    def __post_init__(self) -> None:
        if not self.text or not self.text.strip():
            raise ValueError("ExtractedSentence.text cannot be empty")
        if self.paragraph_number < 1 or self.sentence_number < 1:
            raise ValueError("paragraph_number/sentence_number must be >= 1")

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """A sentence-aware chunk built from one or more consecutive sentences.

    Chunks never split a sentence in half. `*_start`/`*_end` fields let a
    UI (Phase 4) map a chunk back to an exact location in the source file
    for highlighting, even though `text` is the already-joined string used
    for embedding.
    """

    chunk_id: str
    filename: str
    page_start: Optional[int]
    page_end: Optional[int]
    paragraph_start: int
    paragraph_end: int
    sentence_start: int
    sentence_end: int
    text: str
    word_count: int
    sentence_count: int

    def __post_init__(self) -> None:
        if not self.text or not self.text.strip():
            raise ValueError("DocumentChunk.text cannot be empty")
        if self.sentence_count < 1:
            raise ValueError("DocumentChunk must contain at least one sentence")
