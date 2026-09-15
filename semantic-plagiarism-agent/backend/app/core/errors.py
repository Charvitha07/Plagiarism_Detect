"""Domain-specific exceptions.

Kept separate from FastAPI so the NLP pipeline (Phase 1) has zero web
framework dependency and can be unit tested / reused as a plain library.
The API layer (Phase 3) will catch these and map them to HTTP status
codes instead of leaking tracebacks or internal paths to clients.
"""


class DocumentProcessingError(Exception):
    """Base class for all document-processing failures."""


class UnsupportedFileTypeError(DocumentProcessingError):
    """Raised when a file's extension isn't one of the allowed types."""


class CorruptedFileError(DocumentProcessingError):
    """Raised when a file can't be opened/parsed by its format library."""


class EmptyDocumentError(DocumentProcessingError):
    """Raised when extraction succeeds but yields no usable text."""


class EmbeddingBackendUnavailableError(DocumentProcessingError):
    """Raised when a real embedding backend is requested but its
    dependencies (or downloaded model weights) aren't available."""
