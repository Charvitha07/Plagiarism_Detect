"""
Sentence splitting, kept behind a small interface so the extraction
pipeline doesn't care whether sentences come from a regex splitter or a
real NLP library.

Default: `RegexSentenceSplitter` — dependency-free, deterministic, good
enough for well-formed prose (the kind found in reference/submitted
academic documents) and guards a common abbreviation list so "Dr. Rao
studied U.S. history." isn't split into three fragments.

Optional: `SpacySentenceSplitter` — higher quality, used automatically
when `spacy` *and* a downloaded model are available (see
`get_default_splitter`). Not required for the pipeline to run or for
the test suite to pass.
"""
from __future__ import annotations

import re
from typing import List, Protocol


class SentenceSplitter(Protocol):
    def split(self, text: str) -> List[str]:
        ...


# Common abbreviations whose trailing "." should NOT be treated as a
# sentence boundary. Matched case-insensitively at a token boundary.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
    "e.g", "i.e", "u.s", "u.k", "a.m", "p.m", "fig", "eq", "no", "vol",
    "approx", "dept", "univ", "inc", "ltd", "co",
}

_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\(])")
_WHITESPACE_RE = re.compile(r"\s+")


class RegexSentenceSplitter:
    """Splits on ., !, ? followed by whitespace + a capital/number/quote,
    while protecting a small set of common abbreviations from being
    mistaken for sentence boundaries."""

    def split(self, text: str) -> List[str]:
        normalized = _WHITESPACE_RE.sub(" ", text.strip())
        if not normalized:
            return []

        candidates = _SENTENCE_END_RE.split(normalized)
        sentences: List[str] = []
        buffer = ""
        for piece in candidates:
            buffer = f"{buffer} {piece}".strip() if buffer else piece
            if self._ends_on_abbreviation(buffer):
                # Don't close the sentence yet — merge with the next piece.
                continue
            sentences.append(buffer.strip())
            buffer = ""
        if buffer.strip():
            sentences.append(buffer.strip())
        return [s for s in sentences if s]

    @staticmethod
    def _ends_on_abbreviation(fragment: str) -> bool:
        last_token = fragment.split(" ")[-1].rstrip(".").lower()
        return last_token in _ABBREVIATIONS


class SpacySentenceSplitter:
    """Optional higher-quality splitter backed by spaCy's sentencizer.

    Requires `spacy` and a downloaded pipeline (e.g. `en_core_web_sm`).
    Raises a clear error at construction time if either is missing,
    rather than failing confusingly mid-pipeline.
    """

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        try:
            import spacy
        except ImportError as exc:  # pragma: no cover - exercised only when spaCy is absent
            raise RuntimeError(
                "spaCy is not installed. Run: pip install spacy && "
                "python -m spacy download en_core_web_sm"
            ) from exc
        try:
            self._nlp = spacy.load(model_name, exclude=["ner", "lemmatizer"])
        except OSError as exc:  # pragma: no cover - exercised only when model isn't downloaded
            raise RuntimeError(
                f"spaCy model {model_name!r} isn't downloaded. Run: "
                f"python -m spacy download {model_name}"
            ) from exc
        if "senter" not in self._nlp.pipe_names and "parser" not in self._nlp.pipe_names:
            self._nlp.add_pipe("sentencizer")

    def split(self, text: str) -> List[str]:
        doc = self._nlp(text.strip())
        return [s.text.strip() for s in doc.sents if s.text.strip()]


def get_default_splitter() -> SentenceSplitter:
    """Best available splitter: spaCy if it's actually usable, else regex."""
    try:
        return SpacySentenceSplitter()
    except RuntimeError:
        return RegexSentenceSplitter()
