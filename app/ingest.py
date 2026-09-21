"""Index a directory of .md/.txt/.rst files.

    python -m app.ingest ./corpus

doc_type is inferred from the path: anything under a folder whose name contains
"experiment" or "report" is tagged 'experiment', everything else 'docs'.
Unchanged files (same content hash) are skipped; changed files are re-chunked.
"""
import hashlib
import sys
from pathlib import Path

from .chunking import chunk_markdown
from .db import connect, init_schema
from .embeddings import embed_documents

EXTS = {".md", ".txt", ".rst"}


def infer_type(path: Path) -> str:
    p = str(path).lower()
    return "experiment" if ("experiment" in p or "report" in p) else "docs"


def ingest_dir(root: Path) -> None:
    init_schema()
    conn = connect()
    for f in sorted(root.rglob("*")):
        if f.suffix.lower() not in EXTS or not f.is_file():
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        digest = hashlib.sha256(text.encode()).hexdigest()
        rel = str(f.relative_to(root))

        row = conn.execute(
            "SELECT id, content_hash FROM documents WHERE path=%s", (rel,)
        ).fetchone()
        if row and row[1] == digest:
            print(f"skip   {rel}")
            continue

        title, chunks = chunk_markdown(text)
        if not chunks:
            continue
        vectors = embed_documents([f"{c.section}\n{c.content}" for c in chunks])

        with conn.transaction():
            if row:
                conn.execute("DELETE FROM documents WHERE id=%s", (row[0],))
            doc_id = conn.execute(
                "INSERT INTO documents (path, title, doc_type, content_hash) "
                "VALUES (%s,%s,%s,%s) RETURNING id",
                (rel, title, infer_type(f), digest),
            ).fetchone()[0]
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (doc_id, chunk_index, section, content, embedding) "
                    "VALUES (%s,%s,%s,%s,%s)",
                    [(doc_id, i, c.section, c.content, v)
                     for i, (c, v) in enumerate(zip(chunks, vectors))],
                )
        print(f"indexed {rel} ({len(chunks)} chunks)")


if __name__ == "__main__":
    ingest_dir(Path(sys.argv[1] if len(sys.argv) > 1 else "corpus"))
