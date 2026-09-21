# RAG-based ML Knowledge Assistant

Indexes technical docs and experiment reports into Postgres/pgvector, retrieves with hybrid
search (vector + full-text, fused with RRF), and answers with numbered citations.

## Run

```bash
docker compose up -d
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...        # only needed for /ask

python -m app.ingest ./corpus          # re-run any time; unchanged files are skipped
uvicorn app.main:app --reload
```

First ingest downloads the embedding model (~130 MB, cached afterwards).

```bash
# retrieval only, no LLM: see exactly what the model would be given
curl -s localhost:8000/search -H 'content-type: application/json' \
  -d '{"question": "Which augmentation gave the largest gain?"}'

# full answer with citations
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "Which learning rate worked best for ResNet-50 and why not higher?"}'
```

Put your own `.md`/`.txt`/`.rst` files under `corpus/` (folders with "experiment" or "report" in
the path are tagged `experiment`, the rest `docs`), or point ingest at any folder.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q          # needs the docker compose db running
```

Tests use a throwaway `rag_test` database, the offline `EMBED_BACKEND=hash` embedder, and a fake
LLM, so they need no model download or API key. They cover chunking, idempotent re-ingest,
change detection, hybrid retrieval, filters, hostile queries, citation handling, and the HTTP API.

## Layout

| File | Role |
|------|------|
| `app/chunking.py` | Heading-aware chunking; keeps code fences intact; chunks carry their section path |
| `app/ingest.py` | Walks a folder, hashes files, re-embeds only what changed |
| `app/retrieve.py` | Vector + keyword candidates fused with RRF; similarity floor; optional `doc_type` filter |
| `app/answer.py` | Grounded prompt, `[n]` citations, returns only sources actually cited |
| `app/main.py` | FastAPI: `POST /ask`, `POST /search`, `GET /health` |
| `app/embeddings.py` | `st` (real model) or `hash` (offline, for tests only) backends |

## Configuration (env vars)

`DATABASE_URL`, `EMBED_MODEL`, `EMBED_DIM` (must match the model), `LLM_MODEL`,
`CHUNK_WORDS`, `CHUNK_OVERLAP_WORDS`, `MIN_SIMILARITY`.
Changing the embedding model means updating `EMBED_DIM`, dropping the tables, and re-ingesting.

## Known limits / next steps

- **`MIN_SIMILARITY` (0.30) is a starting guess.** Tune it on real questions from your corpus:
  too high drops good chunks, too low lets irrelevant ones through to the LLM.
- No eval harness yet. Build 30-50 real questions with expected source docs and track
  recall@k and citation accuracy before changing chunking or retrieval.
- Not yet built: reranker, PDF/notebook loaders, auth, streaming responses.
