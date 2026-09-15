import pytest

from app.schemas.document import DocumentChunk, ExtractedSentence, SupportedFileType


def test_extracted_sentence_rejects_empty_text():
    with pytest.raises(ValueError):
        ExtractedSentence(filename="a.txt", page_number=None, paragraph_number=1, sentence_number=1, text="   ")


def test_extracted_sentence_rejects_non_positive_numbers():
    with pytest.raises(ValueError):
        ExtractedSentence(filename="a.txt", page_number=None, paragraph_number=0, sentence_number=1, text="Hi.")


def test_extracted_sentence_word_count():
    sentence = ExtractedSentence(
        filename="a.txt", page_number=None, paragraph_number=1, sentence_number=1, text="Four little words here."
    )
    assert sentence.word_count == 4


def test_document_chunk_rejects_empty_text():
    with pytest.raises(ValueError):
        DocumentChunk(
            chunk_id="x",
            filename="a.txt",
            page_start=None,
            page_end=None,
            paragraph_start=1,
            paragraph_end=1,
            sentence_start=1,
            sentence_end=1,
            text="   ",
            word_count=0,
            sentence_count=0,
        )


def test_supported_file_type_from_suffix():
    assert SupportedFileType.from_suffix(".PDF") is SupportedFileType.PDF
    assert SupportedFileType.from_suffix("docx") is SupportedFileType.DOCX


def test_supported_file_type_rejects_unknown_suffix():
    with pytest.raises(ValueError):
        SupportedFileType.from_suffix(".md")
