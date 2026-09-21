"""Embedding backends.

EMBED_BACKEND=st    (default) sentence-transformers model from EMBED_MODEL
EMBED_BACKEND=hash  offline, deterministic bag-of-words hashing. Not semantic,
                    but it lets you run the whole pipeline and the tests with
                    no model download. Do not use it for real answers.
"""
import hashlib
import math
import re
from functools import lru_cache

from .config import EMBED_BACKEND, EMBED_DIM, EMBED_MODEL, QUERY_PREFIX


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


def _hash_embed(text: str) -> list[float]:
    vec = [0.0] * EMBED_DIM
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % EMBED_DIM] += 1.0 if (h >> 20) & 1 else -1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def embed_documents(texts: list[str]) -> list[list[float]]:
    if EMBED_BACKEND == "hash":
        return [_hash_embed(t) for t in texts]
    return _model().encode(texts, normalize_embeddings=True, batch_size=32).tolist()


def embed_query(text: str) -> list[float]:
    if EMBED_BACKEND == "hash":
        return _hash_embed(text)
    return _model().encode(QUERY_PREFIX + text, normalize_embeddings=True).tolist()
