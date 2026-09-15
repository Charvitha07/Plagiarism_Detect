"""
Extracts provenance-tagged sentences from PDF, DOCX, and TXT files.

Every extractor returns `list[ExtractedSentence]` with filename, page
number (None where the format has no pagination concept), paragraph
number, sentence number, and the original sentence text — everything
the chunking/report layers need to point back at "page 3, paragraph 2"
without re-parsing the source file.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from app.core.errors import (
    CorruptedFileError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
)
from app.schemas.document import ExtractedSentence, SupportedFileType
from app.services.sentence_split import RegexSentenceSplitter, SentenceSplitter

_BLANK_LINE_RE = re.compile(r"\n\s*\n+")


def extract_document(
    path: Path,
    splitter: Optional[SentenceSplitter] = None,
    original_filename: Optional[str] = None,
) -> List[ExtractedSentence]:
    """Dispatches to the right extractor based on file extension.

    `original_filename` lets callers pass the user-facing filename even
    when `path` points at a temp file with a generated name.
    """
    splitter = splitter or RegexSentenceSplitter()
    filename = original_filename or path.name

    try:
        file_type = SupportedFileType.from_suffix(path.suffix)
    except ValueError as exc:
        raise UnsupportedFileTypeError(
            f"'{path.suffix}' is not a supported file type. "
            f"Allowed: .pdf, .docx, .txt"
        ) from exc

    if file_type is SupportedFileType.PDF:
        sentences = _extract_pdf(path, filename, splitter)
    elif file_type is SupportedFileType.DOCX:
        sentences = _extract_docx(path, filename, splitter)
    else:
        sentences = _extract_txt(path, filename, splitter)

    if not sentences:
        raise EmptyDocumentError(f"No extractable text found in '{filename}'")
    return sentences


def _extract_pdf(path: Path, filename: str, splitter: SentenceSplitter) -> List[ExtractedSentence]:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyMuPDF is not installed. Run: pip install PyMuPDF") from exc

    try:
        doc = fitz.open(str(path))
    except Exception as exc:  # PyMuPDF raises its own error types
        raise CorruptedFileError(f"Could not open PDF '{filename}': {exc}") from exc

    sentences: List[ExtractedSentence] = []
    paragraph_no = 0
    sentence_no = 0
    try:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            blocks = page.get_text("blocks")
            # Reading order: top-to-bottom, then left-to-right.
            blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))
            for block in blocks:
                block_text = block[4].strip() if len(block) > 4 else ""
                if not block_text:
                    continue
                paragraph_no += 1
                for sent in splitter.split(block_text):
                    sentence_no += 1
                    sentences.append(
                        ExtractedSentence(
                            filename=filename,
                            page_number=page_index + 1,
                            paragraph_number=paragraph_no,
                            sentence_number=sentence_no,
                            text=sent,
                        )
                    )
    except CorruptedFileError:
        raise
    except Exception as exc:
        raise CorruptedFileError(f"Failed while reading PDF '{filename}': {exc}") from exc
    finally:
        doc.close()

    return sentences


def _extract_docx(path: Path, filename: str, splitter: SentenceSplitter) -> List[ExtractedSentence]:
    try:
        from docx import Document as DocxDocument
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("python-docx is not installed. Run: pip install python-docx") from exc

    try:
        doc = DocxDocument(str(path))
    except Exception as exc:
        raise CorruptedFileError(f"Could not open DOCX '{filename}': {exc}") from exc

    sentences: List[ExtractedSentence] = []
    paragraph_no = 0
    sentence_no = 0
    try:
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            paragraph_no += 1
            for sent in splitter.split(text):
                sentence_no += 1
                sentences.append(
                    ExtractedSentence(
                        filename=filename,
                        page_number=None,  # DOCX has no reliable page concept pre-render
                        paragraph_number=paragraph_no,
                        sentence_number=sentence_no,
                        text=sent,
                    )
                )
    except Exception as exc:
        raise CorruptedFileError(f"Failed while reading DOCX '{filename}': {exc}") from exc

    return sentences


def _extract_txt(path: Path, filename: str, splitter: SentenceSplitter) -> List[ExtractedSentence]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CorruptedFileError(f"Could not read TXT '{filename}': {exc}") from exc

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        # Best-effort recovery for non-UTF-8 text files rather than failing outright.
        text = raw.decode("utf-8", errors="replace")

    paragraphs = [p.strip() for p in _BLANK_LINE_RE.split(text) if p.strip()]

    sentences: List[ExtractedSentence] = []
    sentence_no = 0
    for paragraph_no, paragraph in enumerate(paragraphs, start=1):
        for sent in splitter.split(paragraph):
            sentence_no += 1
            sentences.append(
                ExtractedSentence(
                    filename=filename,
                    page_number=None,
                    paragraph_number=paragraph_no,
                    sentence_number=sentence_no,
                    text=sent,
                )
            )
    return sentences
