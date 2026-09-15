from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List
from app.schemas.data_models import EmbeddingResult, SimilarityMatch, SimilarityResult
from app.config import settings

class SimilarityCalculator:
    @staticmethod
    def compare(
        source_doc_id: str,
        target_doc_id: str,
        source_embeddings: List[EmbeddingResult],
        target_embeddings: List[EmbeddingResult],
        threshold: float = settings.SIMILARITY_THRESHOLD
    ) -> SimilarityResult:
        if not source_embeddings or not target_embeddings:
            return SimilarityResult(source_document_id=source_doc_id, target_document_id=target_doc_id, matches=[])

        src_vecs = np.array([res.embedding for res in source_embeddings])
        tgt_vecs = np.array([res.embedding for res in target_embeddings])

        sim_matrix = cosine_similarity(src_vecs, tgt_vecs)

        matches = []
        for i, row in enumerate(sim_matrix):
            for j, score in enumerate(row):
                if score >= threshold:
                    src_chunk = source_embeddings[i].chunk
                    tgt_chunk = target_embeddings[j].chunk
                    matches.append(SimilarityMatch(
                        source_chunk_id=src_chunk.chunk_id,
                        target_chunk_id=tgt_chunk.chunk_id,
                        similarity_score=float(score),
                        source_text=src_chunk.cleaned_text,
                        target_text=tgt_chunk.cleaned_text,
                        source_language=src_chunk.language_code,
                        target_language=tgt_chunk.language_code,
                        source_page_numbers=src_chunk.page_numbers,
                        target_page_numbers=tgt_chunk.page_numbers
                    ))

        matches.sort(key=lambda x: x.similarity_score, reverse=True)
        return SimilarityResult(
            source_document_id=source_doc_id,
            target_document_id=target_doc_id,
            matches=matches
        )