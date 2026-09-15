import uuid
import logging
from pathlib import Path
from typing import List
from app.schemas.data_models import ExtractedDocument, EmbeddingResult, SimilarityResult
from app.ingestion.extractor import DocumentExtractor
from app.preprocessing.cleaner import TextCleaner
from app.preprocessing.segmentation import SentenceSegmenter
from app.preprocessing.chunking import ChunkingEngine
from app.language.detector import LanguageDetector
from app.embeddings.generator import EmbeddingGenerator
from app.similarity.calculator import SimilarityCalculator
from app.config import settings

logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    def __init__(self):
        self.embedder = EmbeddingGenerator()

    def process_file(self, file_path: str) -> List[EmbeddingResult]:
        logger.info(f"Processing {file_path}")
        document_id = Path(file_path).name + "_" + str(uuid.uuid4())[:8]
        
        raw_pages = DocumentExtractor.extract(file_path)
        cleaned_pages = {k: TextCleaner.clean(v) for k, v in raw_pages.items()}
        
        doc = ExtractedDocument(
            document_id=document_id, filename=Path(file_path).name,
            raw_pages=raw_pages, cleaned_pages=cleaned_pages
        )
        
        segments = SentenceSegmenter.segment(raw_pages)
        chunks = ChunkingEngine.create_chunks(
            document_id=doc.document_id, segments=segments,
            chunk_size=settings.CHUNK_SIZE_SENTENCES, chunk_overlap=settings.CHUNK_OVERLAP_SENTENCES
        )
        
        for chunk in chunks:
            lang_code, conf = LanguageDetector.detect(chunk.cleaned_text)
            chunk.language_code = lang_code
            chunk.language_confidence = conf
            
        texts_to_embed = [chunk.cleaned_text for chunk in chunks]
        embeddings = self.embedder.generate(texts_to_embed)
        
        return [EmbeddingResult(chunk=chunk, embedding=emb) for chunk, emb in zip(chunks, embeddings)]

    def compare_documents(self, source_path: str, target_path: str) -> SimilarityResult:
        source_embs = self.process_file(source_path)
        target_embs = self.process_file(target_path)
        
        return SimilarityCalculator.compare(
            source_doc_id=source_embs[0].chunk.document_id if source_embs else source_path,
            target_doc_id=target_embs[0].chunk.document_id if target_embs else target_path,
            source_embeddings=source_embs,
            target_embeddings=target_embs
        )