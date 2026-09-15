# Architecture — Semantic Plagiarism Detection Agent

## What it does

Compares a submitted document against a reference corpus using sentence
embeddings, not exact string matching, so paraphrased or restructured
content still surfaces as a match. Every score is reported alongside its
definition — the system never collapses "semantically similar" into
"plagiarized."

## Pipeline (end to end)

```
Reference doc (PDF/DOCX/TXT)
   │  extraction.py            → ExtractedSentence[] (filename, page, paragraph, sentence, text)
   ▼
chunking.py                    → DocumentChunk[] (sentence-aware, never splits a sentence)
   ▼
embeddings.py                  → normalized vectors (batched, model-configurable)
   ▼
FAISS index (Phase 2)          → persisted, vector_id -> chunk_id mapping in SQLite
   ▼
Submitted doc  ──same pipeline──┐
                                 ▼
                        similarity.py / FAISS search (Phase 2)
                                 ▼
                        aggregation (Phase 2/3): group adjacent matches,
                        classify (exact / lightly modified / paraphrase /
                        partial), compute coverage without double-counting
                                 ▼
                        Analysis + Match rows (Phase 3, SQLite)
                                 ▼
                        FastAPI report endpoint (Phase 3)
                                 ▼
                        React report UI (Phase 4)
```

## Repository layout (target — grows phase by phase)

```
semantic-plagiarism-detector/
├── backend/
│   ├── app/
│   │   ├── core/
│   │   │   ├── config.py          # Settings (env-driven, SPDA_ prefix)
│   │   │   └── errors.py          # Domain exceptions (framework-free)
│   │   ├── schemas/
│   │   │   ├── document.py        # ExtractedSentence, DocumentChunk       [Phase 1]
│   │   │   ├── api.py             # Pydantic request/response models      [Phase 3]
│   │   ├── services/
│   │   │   ├── sentence_split.py  # Regex + optional spaCy splitter        [Phase 1]
│   │   │   ├── extraction.py      # PDF/DOCX/TXT -> ExtractedSentence[]    [Phase 1]
│   │   │   ├── chunking.py        # Sentence-aware chunking                [Phase 1]
│   │   │   ├── embeddings.py      # Batched, normalized embeddings         [Phase 1]
│   │   │   ├── similarity.py      # Cosine similarity / top-k (numpy)      [Phase 1]
│   │   │   ├── vector_index.py    # FAISS index + id<->chunk_id mapping    [Phase 2]
│   │   │   ├── retrieval.py       # Corpus-wide top-k retrieval            [Phase 2]
│   │   │   ├── aggregation.py     # Group matches, classify, coverage %    [Phase 2]
│   │   │   └── reporting.py       # Build the report payload / PDF export  [Phase 3]
│   │   ├── models/                # SQLAlchemy ORM models                  [Phase 3]
│   │   ├── api/                   # FastAPI routers                       [Phase 3]
│   │   └── main.py                # FastAPI app factory                   [Phase 3]
│   ├── tests/                                                              [Phase 1+]
│   ├── data/                      # SQLite file + FAISS index (gitignored)
│   ├── requirements.txt
│   └── pytest.ini
├── frontend/                                                               [Phase 4]
│   ├── src/
│   │   ├── pages/ (Dashboard, Corpus, Analysis, Report, Evaluation)
│   │   ├── components/
│   │   └── api/ (typed fetch client)
│   ├── package.json
│   └── vite.config.ts
├── evaluation/                                                             [Phase 5]
│   ├── dataset/                   # Labeled exact/paraphrase/unrelated/partial examples
│   └── run_eval.py                # Precision/recall/F1/FPR/recall@k/latency
├── docs/
│   └── ARCHITECTURE.md            # this file
├── docker-compose.yml                                                      [Phase 5]
├── .env.example
└── README.md
```

## Database schema (Phase 3)

- **ReferenceDocument**: id, filename, file_hash (dedup), file_type, status
  (pending/processing/indexed/failed), chunk_count, uploaded_at
- **DocumentChunk**: id, reference_document_id (FK), vector_id (FK into
  FAISS), page_start/end, paragraph_start/end, sentence_start/end, text
- **Analysis**: id, submitted_filename, mode (corpus/direct), threshold,
  top_k, status, total_words, total_sentences, matched_words,
  matched_sentences, matched_coverage_pct, avg_similarity, max_similarity,
  matched_section_count, matched_reference_doc_count, created_at
- **Match**: id, analysis_id (FK), submitted_chunk text + span, matched
  reference chunk_id, cosine_similarity, classification (exact / lightly
  modified / paraphrase / partial overlap), section_group_id
- **EvaluationExample** (optional): id, pair_a, pair_b, label, category

The FAISS index itself stays a separate on-disk artifact
(`data/faiss_index/`); SQLite never stores vectors, only the
`vector_id <-> chunk_id` mapping, so the index can be rebuilt from the DB
if it's ever lost.

## Metrics — definitions (never fabricated, always computed)

- **matched coverage %** = matched submitted words / total submitted words × 100
  (matched words are deduplicated across overlapping matched spans before
  counting, per the aggregation step, so coverage can never exceed 100%)
- **semantic similarity**: cosine similarity of two chunk embeddings,
  reported per match — a *distance* measure, not a verdict
- **plagiarism probability**: never reported as a bare number. The system
  surfaces coverage + similarity distribution + classification counts and
  leaves the plagiarism judgment to a human reviewer.
- Classification thresholds (configurable): exact copy (near-1.0
  similarity + high lexical overlap), lightly modified (high similarity +
  some lexical overlap), paraphrase (high similarity + low lexical
  overlap), partial overlap (moderate similarity over part of the span).

## Phase status

| Phase | Contents | Status |
|---|---|---|
| 1 | NLP pipeline: extraction, chunking, embeddings, similarity math, tests | **Done** (this delivery) |
| 2 | FAISS indexing, corpus retrieval, match aggregation, coverage calc | Not started |
| 3 | SQLAlchemy models, FastAPI endpoints, report generation | Not started |
| 4 | React dashboard, analysis UI, interactive report | Not started |
| 5 | Evaluation harness, integration tests, Docker, docs | Not started |

## Why the embedding backend is pluggable

`EmbeddingBackend` is an interface with two implementations:
`SentenceTransformerBackend` (real, lazy-imports `sentence-transformers`)
and `DeterministicHashBackend` (dependency-free, deterministic, used in
this repo's own test suite). This isn't a placeholder — it's how the
model becomes configurable per the spec (MiniLM vs. multilingual MiniLM),
and it's what lets `pytest` run in any environment, including ones
without internet access to Hugging Face, while still exercising every
line of real pipeline logic (batching, normalization, chunk boundaries).
Swap in `SentenceTransformerBackend(settings.embedding_model_name)` for
real inference — see README for the one-line wiring.
