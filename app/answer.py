"""Answer generation with numbered citations.

The LLM backend is pluggable (see config.LLM_BACKEND): Claude via the Anthropic API,
or any OpenAI-compatible server such as a local Ollama.
"""
import os
import re
from dataclasses import dataclass

import anthropic
import httpx

from . import config
from .retrieve import Hit, search

SYSTEM = """You are a knowledge assistant for an ML team's documentation and \
experiment reports.

Rules:
- Answer ONLY from the numbered sources provided. Do not use outside knowledge.
- Cite every factual claim with its source number in square brackets, e.g. [1] or [2][3].
- Quote numbers (metrics, hyperparameters, dates) exactly as written in the sources.
- If sources conflict (e.g. two experiments report different results), say so and cite both.
- If the sources do not contain the answer, say you could not find it in the indexed \
material. Do not guess."""

_client = None


class LLMUnavailable(RuntimeError):
    """The LLM isn't configured or can't be reached (-> HTTP 503)."""


class LLMRequestError(RuntimeError):
    """The LLM was reached but the request failed (-> HTTP 502)."""


def _llm():
    global _client
    if _client is None:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise LLMUnavailable("ANTHROPIC_API_KEY is not set")
        _client = anthropic.Anthropic()
    return _client


def _generate_anthropic(system: str, user: str) -> str:
    try:
        msg = _llm().messages.create(
            model=config.LLM_MODEL or "claude-sonnet-5",
            max_tokens=1000,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
    except anthropic.APIError as e:
        raise LLMRequestError(type(e).__name__) from e
    return "".join(b.text for b in msg.content if b.type == "text")


def _generate_openai(system: str, user: str) -> str:
    if not config.LLM_MODEL:
        raise LLMUnavailable("LLM_MODEL is not set (for Ollama, e.g. llama3.2)")
    url = config.LLM_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {config.LLM_API_KEY}"} if config.LLM_API_KEY else {}
    body = {
        "model": config.LLM_MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "max_tokens": 1000,
        "temperature": 0,
    }
    try:
        # Generous timeout: a local model on CPU can take a while, especially on the
        # first request while it loads into memory.
        r = httpx.post(url, json=body, headers=headers, timeout=300)
    except httpx.ConnectError as e:
        raise LLMUnavailable(
            f"cannot reach the LLM server at {config.LLM_BASE_URL}. Is it running?") from e
    except httpx.HTTPError as e:
        raise LLMRequestError(type(e).__name__) from e
    if r.status_code != 200:
        raise LLMRequestError(f"HTTP {r.status_code}: {r.text[:200]}")
    try:
        return r.json()["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, ValueError) as e:
        raise LLMRequestError("unexpected response format") from e


def _generate(system: str, user: str) -> str:
    if config.LLM_BACKEND == "anthropic":
        text = _generate_anthropic(system, user)
    elif config.LLM_BACKEND == "openai":
        text = _generate_openai(system, user)
    else:
        raise LLMUnavailable(
            f"unknown LLM_BACKEND {config.LLM_BACKEND!r} (use 'anthropic' or 'openai')")
    if not text.strip():
        raise LLMRequestError("the model returned an empty answer")
    return text


@dataclass
class Answer:
    answer: str
    citations: list[dict]


def _format_context(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{i}] {h.title} | {h.path} | section: {h.section}\n{h.content}"
        for i, h in enumerate(hits, 1)
    )


def ask(question: str, top_k: int = 6, doc_type: str | None = None) -> Answer:
    hits = search(question, top_k=top_k, doc_type=doc_type)
    if not hits:
        return Answer("I couldn't find anything relevant in the indexed material.", [])

    text = _generate(
        SYSTEM, f"Sources:\n\n{_format_context(hits)}\n\nQuestion: {question}")

    # Return only sources the model actually cited, and drop citation numbers
    # that don't map to a real source.
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(hits)})
    citations = [
        {"id": n, "path": hits[n - 1].path, "title": hits[n - 1].title,
         "section": hits[n - 1].section, "snippet": hits[n - 1].content[:300]}
        for n in cited
    ]
    return Answer(text, citations)
