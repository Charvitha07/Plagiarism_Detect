# Semantic Plagiarism Detection Agent

Detects similarity by **meaning**, not exact wording — built to catch
paraphrased or restructured content that exact-match plagiarism checkers
miss. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full
system design and phase roadmap.

**Status: Phase 1 of 5 complete** — the NLP pipeline (extraction,
chunking, embeddings, similarity math) with a full test suite. Phases
2–5 (FAISS retrieval, FastAPI + DB, React UI, evaluation harness +
Docker) are designed in `docs/ARCHITECTURE.md` but not yet implemented.

## Phase 1 — what's here

- `app/services/extraction.py` — pulls sentence-level text out of PDF,
  DOCX, and TXT files, tagged with filename/page/paragraph/sentence
- `app/services/sentence_split.py` — dependency-free regex sentence
  splitter (default) with an optional spaCy backend
- `app/services/chunking.py` — groups sentences into embedding-ready
  chunks without ever splitting a sentence across two chunks
- `app/services/embeddings.py` — batches + L2-normalizes embeddings
  behind a swappable backend interface
- `app/services/similarity.py` — cosine similarity + top-k retrieval
  (pure numpy; this is also the correctness oracle for Phase 2's FAISS
  layer)
- `backend/tests/` — 63 tests covering all of the above, including a
  full extraction → chunking → embedding integration test per file type

## Setup

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example ../.env
```

`requirements.txt` includes `sentence-transformers` and `torch` for real
embeddings — those are **not required to run the test suite** (see
below). Installing them requires internet access to download model
weights from Hugging Face on first use.

## Running the tests

```bash
cd backend
python3 -m pytest -q                         # 63 tests, no ML deps needed
python3 -m pytest --cov=app --cov-report=term-missing   # with coverage (~90%)
```

The test suite runs against `DeterministicHashBackend`, a dependency-free
stand-in embedder (see `app/services/embeddings.py`), specifically so CI
and this sandbox can validate every pipeline stage — batching,
normalization, chunk boundaries, provenance tracking — without needing
torch installed or network access to Hugging Face.

## Using it with real embeddings

Once `sentence-transformers`/`torch` are installed and you have internet
access for the first model download:

```python
from app.core.config import get_settings
from app.services.embeddings import EmbeddingService, SentenceTransformerBackend

settings = get_settings()
backend = SentenceTransformerBackend(settings.embedding_model_name)  # all-MiniLM-L6-v2 by default
service = EmbeddingService(backend, batch_size=settings.embedding_batch_size)

vectors = service.embed_chunks(chunks)  # chunks from chunking.chunk_sentences(...)
```

Switch models by setting `SPDA_EMBEDDING_MODEL_NAME` in `.env` — e.g. to
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` for
multilingual documents.

## Quick end-to-end example

```python
from pathlib import Path
from app.services.extraction import extract_document
from app.services.chunking import chunk_sentences
from app.services.embeddings import EmbeddingService, DeterministicHashBackend

sentences = extract_document(Path("some_reference.pdf"))
chunks = chunk_sentences(sentences, min_words=20, max_words=80, overlap_sentences=1)

service = EmbeddingService(DeterministicHashBackend(dimension=384))  # swap for SentenceTransformerBackend
vectors = service.embed_chunks(chunks)
print(vectors.shape)  # (num_chunks, 384)
```

## Next: Phase 2

FAISS indexing over `DocumentChunk` embeddings, a persistent
`vector_id <-> chunk_id` mapping, corpus-wide top-k retrieval, adjacent
match grouping, and the coverage-percentage calculation — all designed
in `docs/ARCHITECTURE.md`, ready to build next.
