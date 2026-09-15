import logging
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_BATCH_SIZE: int = 32
    CHUNK_SIZE_SENTENCES: int = 5
    CHUNK_OVERLAP_SENTENCES: int = 2
    SIMILARITY_THRESHOLD: float = 0.75

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()

# Centralized Logger Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)