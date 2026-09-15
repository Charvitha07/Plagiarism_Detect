from pydantic import BaseModel
from typing import List, Dict

class TextSegment(BaseModel):
    original_text: str
    cleaned_text: str
    start_char: int
    end_char: int
    page_number: int
    sentence_index: int

class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    original_text: str
    cleaned_text: str
    page_numbers: List[int]
    sentences: List[TextSegment]
    language_code: str = "unknown"
    language_confidence: float = 0.0

class ExtractedDocument(BaseModel):
    document_id: str
    filename: str
    raw_pages: Dict[int, str]
    cleaned_pages: Dict[int, str]

class EmbeddingResult(BaseModel):
    chunk: DocumentChunk
    embedding: List[float]

class SimilarityMatch(BaseModel):
    source_chunk_id: str
    target_chunk_id: str
    similarity_score: float
    source_text: str
    target_text: str
    source_language: str
    target_language: str
    source_page_numbers: List[int]
    target_page_numbers: List[int]

class SimilarityResult(BaseModel):
    source_document_id: str
    target_document_id: str
    matches: List[SimilarityMatch]