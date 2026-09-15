from __future__ import annotations

from pathlib import Path

import pytest

from app.schemas.document import ExtractedSentence


@pytest.fixture
def sample_txt(tmp_path: Path) -> Path:
    content = (
        "Machine learning models learn patterns from data. "
        "They generalize to unseen examples when trained well.\n\n"
        "Dr. Rao published a paper on U.S. climate policy. "
        "It received wide attention."
    )
    path = tmp_path / "sample.txt"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def empty_txt(tmp_path: Path) -> Path:
    path = tmp_path / "empty.txt"
    path.write_text("   \n\n   ", encoding="utf-8")
    return path


@pytest.fixture
def sample_docx(tmp_path: Path) -> Path:
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_paragraph(
        "Neural networks are inspired by biological brains. "
        "They consist of layers of interconnected nodes."
    )
    document.add_paragraph(
        "Backpropagation adjusts weights using gradients. "
        "This process is repeated over many epochs."
    )
    path = tmp_path / "sample.docx"
    document.save(str(path))
    return path


@pytest.fixture
def corrupted_docx(tmp_path: Path) -> Path:
    path = tmp_path / "corrupted.docx"
    path.write_bytes(b"this is not a real docx file")
    return path


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Semantic search retrieves passages by meaning, not exact words. "
        "It relies on dense vector embeddings.",
        fontsize=11,
    )
    page.insert_text(
        (72, 140),
        "Cosine similarity measures the angle between two vectors. "
        "A score near one means the vectors point in a similar direction.",
        fontsize=11,
    )
    path = tmp_path / "sample.pdf"
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def corrupted_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "corrupted.pdf"
    path.write_bytes(b"%PDF-1.4 this is not a valid pdf body")
    return path


@pytest.fixture
def unsupported_file(tmp_path: Path) -> Path:
    path = tmp_path / "notes.md"
    path.write_text("# not a supported type", encoding="utf-8")
    return path


def make_sentence(
    text: str,
    *,
    filename: str = "doc.txt",
    page_number: int | None = None,
    paragraph_number: int = 1,
    sentence_number: int = 1,
) -> ExtractedSentence:
    """Test helper for building an ExtractedSentence without the ceremony."""
    return ExtractedSentence(
        filename=filename,
        page_number=page_number,
        paragraph_number=paragraph_number,
        sentence_number=sentence_number,
        text=text,
    )
