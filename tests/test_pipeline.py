import pytest
from fastapi.testclient import TestClient

from app import answer as answer_mod
from app.chunking import chunk_markdown
from app.db import connect
from app.ingest import ingest_dir
from app.retrieve import search
from tests.conftest import FakeLLM


# ---------- chunking ----------
def test_chunks_carry_heading_path():
    _, chunks = chunk_markdown("# A\n\n## B\ntext one\n\n## C\ntext two\n")
    assert [c.section for c in chunks] == ["A > B", "A > C"]


def test_code_fence_is_not_split():
    md = "# T\n\n```\nline1\n\nline2 after blank\n```\n"
    _, chunks = chunk_markdown(md)
    assert len(chunks) == 1 and "line2 after blank" in chunks[0].content


def test_long_section_splits_with_overlap(monkeypatch):
    from app import chunking
    monkeypatch.setattr(chunking, "CHUNK_WORDS", 30)
    monkeypatch.setattr(chunking, "CHUNK_OVERLAP_WORDS", 5)
    paras = "\n\n".join(" ".join(f"w{i}_{j}" for j in range(20)) for i in range(4))
    _, chunks = chunk_markdown("# T\n\n" + paras)
    assert len(chunks) > 1
    assert chunks[0].content.split()[-1] in chunks[1].content  # overlap carried forward


# ---------- ingestion ----------
def test_ingest_is_idempotent_and_detects_changes(indexed):
    def count(): 
        with connect() as c:
            return c.execute("SELECT count(*) FROM chunks").fetchone()[0]
    n = count()
    ingest_dir(indexed)                      # unchanged -> no duplicates
    assert count() == n
    f = indexed / "docs/training-guide.md"
    f.write_text(f.read_text() + "\n## Extra\nNew section about gradient clipping.\n")
    ingest_dir(indexed)                      # changed -> re-chunked, old chunks replaced
    assert count() == n + 1
    assert search("gradient clipping", top_k=1)[0].section.endswith("Extra")


# ---------- retrieval ----------
def test_finds_right_chunk(indexed):
    top = search("which augmentation gave the largest gain", top_k=3)[0]
    assert top.path == "experiments/exp-051-augmentation.md"


def test_exact_token_match(indexed):
    assert search("--keep-last", top_k=1)[0].section.endswith("Checkpointing")


def test_doc_type_filter(indexed):
    hits = search("CutMix precision checkpoints", top_k=10, doc_type="docs")
    assert hits and all(h.path.startswith("docs/") for h in hits)


def test_nonsense_query_returns_nothing(indexed):
    assert search("zzzzqqq xxyyzz", top_k=5) == []


@pytest.mark.parametrize("q", ["", "   ", "!!! ??? |||", "'; DROP TABLE chunks; --", "a & b | (c)"])
def test_hostile_queries_do_not_crash(indexed, q):
    search(q or "x", top_k=3)
    with connect() as c:   # and the table is still there
        assert c.execute("SELECT count(*) FROM chunks").fetchone()[0] > 0


# ---------- answering (fake LLM) ----------
def test_citations_map_to_real_sources_and_bogus_ones_are_dropped(indexed, monkeypatch):
    fake = FakeLLM("CutMix gave the largest gain [1]. Also something invented [9].")
    monkeypatch.setattr(answer_mod, "_llm", lambda: fake)
    res = answer_mod.ask("which augmentation gave the largest gain")
    assert [c["id"] for c in res.citations] == [1]
    assert res.citations[0]["path"] == "experiments/exp-051-augmentation.md"
    prompt = fake.calls[0]["messages"][0]["content"]
    assert "[1]" in prompt and "CutMix" in prompt          # sources really were sent


def test_no_hits_skips_llm_entirely(indexed, monkeypatch):
    fake = FakeLLM("should never be called")
    monkeypatch.setattr(answer_mod, "_llm", lambda: fake)
    res = answer_mod.ask("zzzzqqq xxyyzz")
    assert fake.calls == [] and res.citations == []
    assert "couldn't find" in res.answer


# ---------- HTTP API ----------
def test_api(indexed, monkeypatch):
    from app.main import app
    monkeypatch.setattr(answer_mod, "_llm", lambda: FakeLLM("Use bf16 [1]."))
    client = TestClient(app)
    assert client.get("/health").json() == {"ok": True}
    r = client.post("/ask", json={"question": "how do I enable bf16 mixed precision"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Use bf16 [1]." and body["citations"][0]["path"] == "docs/training-guide.md"
    assert client.post("/ask", json={"question": "hi"}).status_code == 422       # too short
    assert client.post("/ask", json={"question": "valid question", "top_k": 99}).status_code == 422


def test_missing_api_key_is_a_clean_503(indexed, monkeypatch):
    from app.main import app
    monkeypatch.setattr(answer_mod, "_client", None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = TestClient(app).post("/ask", json={"question": "how do I enable bf16 mixed precision"})
    assert r.status_code == 503 and "ANTHROPIC_API_KEY" in r.json()["detail"]


def test_search_endpoint_needs_no_llm(indexed):
    from app.main import app
    r = TestClient(app).post("/search", json={"question": "keep-last checkpoints"})
    hits = r.json()["hits"]
    assert r.status_code == 200 and hits[0]["section"].endswith("Checkpointing")
