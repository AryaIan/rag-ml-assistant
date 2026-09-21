"""Hybrid retrieval: pgvector cosine + Postgres full-text, fused with RRF.

Vector search handles paraphrase; full-text catches exact tokens that
embeddings blur (metric names, flag names, run IDs, error strings), which
are common in ML docs and experiment reports.
"""
import re
from dataclasses import dataclass

from .config import MIN_SIMILARITY
from .db import connect
from .embeddings import embed_query

RRF_K = 60


@dataclass
class Hit:
    chunk_id: int
    path: str
    title: str
    section: str
    content: str
    score: float
    similarity: float
    lexical_match: bool


def _or_tsquery(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9_]+", text)
    return " | ".join(words)


def search(question: str, top_k: int = 6, doc_type: str | None = None,
           candidates: int = 30) -> list[Hit]:
    qvec = embed_query(question)
    tsq = _or_tsquery(question)
    type_clause = "AND d.doc_type = %(doc_type)s" if doc_type else ""

    sql = f"""
    WITH vec AS (
        SELECT c.id, ROW_NUMBER() OVER (ORDER BY c.embedding <=> %(q)s::vector) AS r
        FROM chunks c JOIN documents d ON d.id = c.doc_id
        WHERE TRUE {type_clause}
        ORDER BY c.embedding <=> %(q)s::vector
        LIMIT %(cand)s
    ),
    lex AS (
        SELECT c.id, ROW_NUMBER() OVER (ORDER BY ts_rank_cd(c.tsv, tq) DESC) AS r
        FROM chunks c JOIN documents d ON d.id = c.doc_id,
             to_tsquery('english', %(tsq)s) tq
        WHERE c.tsv @@ tq {type_clause}
        ORDER BY ts_rank_cd(c.tsv, tq) DESC
        LIMIT %(cand)s
    )
    SELECT c.id, d.path, d.title, c.section, c.content,
           COALESCE(1.0 / (%(k)s + vec.r), 0) + COALESCE(1.0 / (%(k)s + lex.r), 0) AS score,
           1 - (c.embedding <=> %(q)s::vector) AS similarity,
           lex.id IS NOT NULL AS lexical_match
    FROM chunks c
    JOIN documents d ON d.id = c.doc_id
    LEFT JOIN vec ON vec.id = c.id
    LEFT JOIN lex ON lex.id = c.id
    WHERE vec.id IS NOT NULL OR lex.id IS NOT NULL
    ORDER BY score DESC
    LIMIT %(top_k)s
    """
    params = {"q": qvec, "tsq": tsq or "placeholder", "cand": candidates,
              "k": RRF_K, "top_k": top_k, "doc_type": doc_type}
    with connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    hits = [Hit(*r) for r in rows]
    return [h for h in hits if h.lexical_match or h.similarity >= MIN_SIMILARITY]
