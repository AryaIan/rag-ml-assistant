import psycopg
from pgvector.psycopg import register_vector

from .config import DATABASE_URL, EMBED_DIM

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          SERIAL PRIMARY KEY,
    path        TEXT UNIQUE NOT NULL,
    title       TEXT NOT NULL,
    doc_type    TEXT NOT NULL,          -- 'docs' | 'experiment' | ...
    content_hash TEXT NOT NULL,
    ingested_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    doc_id      INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    section     TEXT NOT NULL,          -- heading path, e.g. "Results > Ablations"
    content     TEXT NOT NULL,
    embedding   vector({EMBED_DIM}) NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (
                    to_tsvector('english', section || ' ' || content)
                ) STORED
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks (doc_id);
"""


def connect() -> psycopg.Connection:
    conn = psycopg.connect(DATABASE_URL, autocommit=True)
    register_vector(conn)
    return conn


def init_schema() -> None:
    # The extension must exist before register_vector can find the type.
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        conn.execute(SCHEMA)
