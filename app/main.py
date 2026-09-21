from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .answer import LLMRequestError, LLMUnavailable, ask
from .retrieve import search

app = FastAPI(title="ML Knowledge Assistant")


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(6, ge=1, le=20)
    doc_type: str | None = Field(None, description="'docs' or 'experiment'")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/search")
def search_endpoint(req: AskRequest):
    """Retrieval only (no LLM). Use this to debug why an answer was good or bad."""
    hits = search(req.question, top_k=req.top_k, doc_type=req.doc_type)
    return {"hits": [
        {"path": h.path, "section": h.section, "score": round(h.score, 4),
         "similarity": round(h.similarity, 3), "lexical_match": h.lexical_match,
         "content": h.content}
        for h in hits
    ]}


@app.post("/ask")
def ask_endpoint(req: AskRequest):
    try:
        result = ask(req.question, top_k=req.top_k, doc_type=req.doc_type)
    except LLMUnavailable as e:
        raise HTTPException(503, f"LLM not configured: {e}")
    except LLMRequestError as e:
        raise HTTPException(502, f"LLM request failed: {e}")
    return {"answer": result.answer, "citations": result.citations}
