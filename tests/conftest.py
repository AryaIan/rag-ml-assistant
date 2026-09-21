"""Tests run against a real Postgres+pgvector (a throwaway `rag_test` database) with
the offline hash embedder and a fake LLM, so they need no model download or API key.

    docker compose up -d && pytest -q
"""
import os
from types import SimpleNamespace

# Must be set before any app module is imported (config reads env at import time).
ADMIN_URL = os.getenv("TEST_ADMIN_URL", "postgresql://rag:rag@localhost:5432/postgres")
os.environ["DATABASE_URL"] = ADMIN_URL.rsplit("/", 1)[0] + "/rag_test"
os.environ["EMBED_BACKEND"] = "hash"

import psycopg
import pytest

CORPUS = {
    "docs/training-guide.md": (
        "# Training Guide\n\n## Mixed precision\n"
        "Enable bf16 with `--precision bf16`. fp16 requires loss scaling.\n\n"
        "## Checkpointing\nCheckpoints are written every 1000 steps. Set `--keep-last 3`.\n"
    ),
    "experiments/exp-051-augmentation.md": (
        "# Experiment 051\n\n## Results\n| augmentation | top-1 acc |\n|---|---|\n"
        "| none | 80.9 |\n| CutMix | 83.1 |\n\n## Conclusion\n"
        "CutMix gave the largest gain (+2.2 points).\n"
    ),
}


@pytest.fixture(scope="session", autouse=True)
def database():
    with psycopg.connect(ADMIN_URL, autocommit=True) as conn:
        conn.execute("DROP DATABASE IF EXISTS rag_test WITH (FORCE)")
        conn.execute("CREATE DATABASE rag_test")
    yield
    with psycopg.connect(ADMIN_URL, autocommit=True) as conn:
        conn.execute("DROP DATABASE IF EXISTS rag_test WITH (FORCE)")


@pytest.fixture
def corpus_dir(tmp_path):
    for rel, text in CORPUS.items():
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
    return tmp_path


@pytest.fixture
def indexed(corpus_dir):
    from app.db import connect, init_schema
    from app.ingest import ingest_dir
    init_schema()
    with connect() as c:
        c.execute("TRUNCATE documents CASCADE")
    ingest_dir(corpus_dir)
    return corpus_dir


class FakeLLM:
    """Stands in for anthropic.Anthropic. Records the prompt and returns canned text."""
    def __init__(self, reply):
        self.reply, self.calls = reply, []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.reply)])
