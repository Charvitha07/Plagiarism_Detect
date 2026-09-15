from app.schemas.data_models import DocumentChunk, TextSegment

class ChunkingEngine:
    @staticmethod
    def create_chunks(document_id: str, segments: list[TextSegment], chunk_size: int, chunk_overlap: int) -> list[DocumentChunk]:
        chunks = []
        if not segments:
            return chunks
        
        step = max(1, chunk_size - chunk_overlap)
        for i in range(0, len(segments), step):
            window = segments[i:i + chunk_size]
            if not window:
                break
                
            original_text = " ".join([seg.original_text for seg in window])
            cleaned_text = " ".join([seg.cleaned_text for seg in window])
            page_numbers = sorted(list(set(seg.page_number for seg in window)))
            
            chunk = DocumentChunk(
                chunk_id=f"{document_id}_chunk_{len(chunks)}",
                document_id=document_id,
                original_text=original_text,
                cleaned_text=cleaned_text,
                page_numbers=page_numbers,
                sentences=window
            )
            chunks.append(chunk)
            
            if i + chunk_size >= len(segments):
                break
                
        return chunks