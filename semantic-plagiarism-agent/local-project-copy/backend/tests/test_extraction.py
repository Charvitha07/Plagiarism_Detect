import pytest

from app.core.errors import (
    CorruptedFileError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
)
from app.services.extraction import extract_document


class TestTxtExtraction:
    def test_extracts_sentences_with_provenance(self, sample_txt):
        sentences = extract_document(sample_txt)

        assert len(sentences) == 4
        assert all(s.filename == "sample.txt" for s in sentences)
        assert all(s.page_number is None for s in sentences)
        # Two paragraphs in the fixture.
        assert {s.paragraph_number for s in sentences} == {1, 2}
        # Sentence numbers increase monotonically across the whole doc.
        assert [s.sentence_number for s in sentences] == [1, 2, 3, 4]

    def test_abbreviation_not_split(self, sample_txt):
        sentences = extract_document(sample_txt)
        texts = [s.text for s in sentences]
        assert any(t.startswith("Dr. Rao published") and "U.S." in t for t in texts)

    def test_empty_txt_raises(self, empty_txt):
        with pytest.raises(EmptyDocumentError):
            extract_document(empty_txt)

    def test_non_utf8_bytes_do_not_crash(self, tmp_path):
        path = tmp_path / "latin1.txt"
        path.write_bytes("Café résumé naïve.".encode("latin-1"))
        # Should not raise, even though the file isn't valid UTF-8.
        sentences = extract_document(path)
        assert len(sentences) >= 1


class TestDocxExtraction:
    def test_extracts_sentences_with_provenance(self, sample_docx):
        sentences = extract_document(sample_docx)

        assert len(sentences) == 4
        assert all(s.filename == "sample.docx" for s in sentences)
        assert all(s.page_number is None for s in sentences)
        assert {s.paragraph_number for s in sentences} == {1, 2}
        assert [s.sentence_number for s in sentences] == [1, 2, 3, 4]

    def test_corrupted_docx_raises(self, corrupted_docx):
        with pytest.raises(CorruptedFileError):
            extract_document(corrupted_docx)


class TestPdfExtraction:
    def test_extracts_sentences_with_page_numbers(self, sample_pdf):
        sentences = extract_document(sample_pdf)

        assert len(sentences) == 4
        assert all(s.filename == "sample.pdf" for s in sentences)
        assert all(s.page_number == 1 for s in sentences)
        assert [s.sentence_number for s in sentences] == [1, 2, 3, 4]

    def test_corrupted_pdf_raises(self, corrupted_pdf):
        with pytest.raises(CorruptedFileError):
            extract_document(corrupted_pdf)


class TestDispatchAndErrors:
    def test_unsupported_extension_raises(self, unsupported_file):
        with pytest.raises(UnsupportedFileTypeError):
            extract_document(unsupported_file)

    def test_original_filename_override(self, sample_txt):
        sentences = extract_document(sample_txt, original_filename="essay_v2.txt")
        assert all(s.filename == "essay_v2.txt" for s in sentences)
