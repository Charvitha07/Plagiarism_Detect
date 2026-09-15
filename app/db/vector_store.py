import uuid
import logging
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from app.schemas.data_models import EmbeddingResult
from app.config import settings

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        # Using local disk storage for development (no Docker required)
        self.client = QdrantClient(path="./qdrant_storage")
        self.collection_name = "plagiarism_chunks"
        self.vector_size = 384  # Size for MiniLM-L12-v2
        self._ensure_collection()

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            logger.info(f"Creating Qdrant collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

    def store_embeddings(self, embeddings: List[EmbeddingResult]):
        if not embeddings:
            return
            
        points = []
        for emb in embeddings:
            # We store the chunk data as the payload so we can retrieve it upon a match
            payload = emb.chunk.model_dump()
            points.append(PointStruct(
                id=str(uuid.uuid4()), 
                vector=emb.embedding, 
                payload=payload
            ))
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info(f"Stored {len(points)} chunks in vector database.")

    def search_similar(self, embedding: List[float], limit: int = 3, threshold: float = settings.SIMILARITY_THRESHOLD):
        # We now use `query_points` and `query` instead of the deprecated `search` syntax
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            limit=limit,
            score_threshold=threshold
        )
        # query_points returns a response object, so we extract the .points array
        return results.points