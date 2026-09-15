"""
Application configuration.

All tunables live here and are overridable via environment variables
(prefix SPDA_) or a `.env` file, per the project's `.env.example`.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

# Known embedding models and their output dimensionality. Used to validate
# configuration and to size the FAISS index (Phase 2) without having to
# load the model first.
KNOWN_EMBEDDING_MODELS: dict[str, int] = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
}


class Settings(BaseSettings):
    # --- Embeddings ---
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32

    # --- Chunking ---
    # Chunks are built by greedily grouping whole sentences (never splitting
    # a sentence) until they reach roughly `max_chunk_words`, but a chunk is
    # only closed early once it already holds at least `min_chunk_words`.
    min_chunk_words: int = 20
    max_chunk_words: int = 80
    chunk_overlap_sentences: int = 1

    # --- Retrieval / analysis (used from Phase 2 onward) ---
    default_top_k: int = 5
    similarity_threshold: float = 0.75

    # --- File handling ---
    max_upload_size_mb: int = 25
    allowed_extensions: tuple[str, ...] = (".pdf", ".docx", ".txt")

    # --- Storage (used from Phase 2/3 onward) ---
    database_url: str = "sqlite:///./data/plagiarism.db"
    faiss_index_dir: str = "./data/faiss_index"

    model_config = SettingsConfigDict(
        env_prefix="SPDA_",
        env_file=".env",
        extra="ignore",
    )

    @property
    def embedding_dimension(self) -> int:
        """Best-known dimension for the configured model.

        Falls back to the MiniLM default (384) for models we don't have a
        static mapping for; the real dimension is always re-read from the
        loaded SentenceTransformer at runtime, this is only used to
        pre-size things (e.g. a FAISS index) before the model is loaded.
        """
        return KNOWN_EMBEDDING_MODELS.get(self.embedding_model_name, 384)


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor — import and call this, don't instantiate
    Settings() directly, so the whole app shares one configuration object."""
    return Settings()
