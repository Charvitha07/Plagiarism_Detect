import torch
import logging
from typing import List
from sentence_transformers import SentenceTransformer
from app.config import settings

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingGenerator, cls).__new__(cls)
            cls._instance._init_model()
        return cls._instance

    def _init_model(self):
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Loading embedding model '{settings.MODEL_NAME}' on {device}")
        self.model = SentenceTransformer(settings.MODEL_NAME, device=device)
        logger.info("Model loaded successfully.")

    def generate(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        # Pre-normalized embeddings guarantee geometric validity against pure Cosine Distance algorithms
        embeddings = self.model.encode(
            texts, 
            batch_size=settings.EMBEDDING_BATCH_SIZE, 
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings.tolist()