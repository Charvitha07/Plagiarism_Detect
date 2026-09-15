import pytest
from app.pipeline.orchestrator import PipelineOrchestrator
from app.ingestion.extractor import ExtractionError

@pytest.fixture(scope="module")
def pipeline():
    return PipelineOrchestrator()

def test_english_exact(pipeline):
    res = pipeline.compare_documents("sample_data/english.txt", "sample_data/english.txt")
    assert len(res.matches) > 0
    assert res.matches[0].similarity_score > 0.99

def test_english_paraphrase(pipeline):
    res = pipeline.compare_documents("sample_data/english.txt", "sample_data/paraphrase.txt")
    assert len(res.matches) > 0
    assert res.matches[0].similarity_score > 0.75

def test_english_hindi(pipeline):
    res = pipeline.compare_documents("sample_data/english.txt", "sample_data/hindi.txt")
    assert len(res.matches) > 0
    assert res.matches[0].similarity_score > 0.75

def test_english_telugu(pipeline):
    res = pipeline.compare_documents("sample_data/english.txt", "sample_data/telugu.txt")
    assert len(res.matches) > 0
    assert res.matches[0].similarity_score > 0.70

def test_unrelated(pipeline):
    res = pipeline.compare_documents("sample_data/english.txt", "sample_data/unrelated.txt")
    assert len(res.matches) == 0

def test_docx_and_pdf_extraction(pipeline):
    res_docx = pipeline.compare_documents("sample_data/english.txt", "sample_data/document.docx")
    assert len(res_docx.matches) > 0
    
    res_pdf = pipeline.process_file("sample_data/dummy.pdf")
    assert "Hello World" in res_pdf[0].chunk.cleaned_text

def test_empty_file_handling(pipeline):
    with open("sample_data/empty.txt", "w") as f:
        f.write("   \n   ")
    res = pipeline.process_file("sample_data/empty.txt")
    assert len(res) == 0

def test_corrupted_file(pipeline):
    with open("sample_data/corrupt.pdf", "w") as f:
        f.write("not a pdf")
    with pytest.raises(ExtractionError):
        pipeline.process_file("sample_data/corrupt.pdf")
        
def test_short_and_unicode(pipeline):
    with open("sample_data/unicode.txt", "w", encoding="utf-8") as f:
        f.write("H\u0065llo world \u200b")
    res = pipeline.process_file("sample_data/unicode.txt")
    assert len(res) > 0 
    assert res[0].chunk.language_code in ["unknown", "en"]  # Langdetect might figure out it's English  # Safely handles short/ambiguous languages natively