import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "st")  # "st" or "hash" (offline/testing)
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))  # must match EMBED_MODEL
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge convention
# Which service writes the final answer:
#   "anthropic": Claude API (needs ANTHROPIC_API_KEY and account credit)
#   "openai":    any OpenAI-compatible server. Ollama (free, runs on your own computer),
#                Groq, Gemini's compatibility endpoint, etc.
LLM_BACKEND = os.getenv("LLM_BACKEND", "anthropic")
LLM_MODEL = os.getenv("LLM_MODEL", "")  # anthropic: defaults to claude-sonnet-5; openai: required
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")  # default = Ollama
LLM_API_KEY = os.getenv("LLM_API_KEY", "")  # not needed for Ollama
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "220"))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "40"))
# A chunk is kept only if it matched the keyword query OR its cosine similarity to
# the question is at least this. Tune on your own eval set; ~0.3-0.5 is typical
# for bge-small. Without a floor, vector search always returns *something*.
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.30"))
