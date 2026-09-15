from app.core.config import Settings


def test_defaults_are_sane():
    settings = Settings(_env_file=None)
    assert settings.embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.min_chunk_words < settings.max_chunk_words
    assert settings.allowed_extensions == (".pdf", ".docx", ".txt")


def test_env_prefix_overrides_defaults(monkeypatch):
    monkeypatch.setenv("SPDA_EMBEDDING_BATCH_SIZE", "64")
    monkeypatch.setenv("SPDA_MAX_CHUNK_WORDS", "120")
    settings = Settings(_env_file=None)
    assert settings.embedding_batch_size == 64
    assert settings.max_chunk_words == 120


def test_embedding_dimension_known_model():
    settings = Settings(_env_file=None, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2")
    assert settings.embedding_dimension == 384


def test_embedding_dimension_unknown_model_falls_back():
    settings = Settings(_env_file=None, embedding_model_name="some/custom-model")
    assert settings.embedding_dimension == 384
