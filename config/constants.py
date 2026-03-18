import os

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 100
THRESHOLD_SIMILARITY = 0.85
EMBED_MODEL_ENGLISH = os.getenv(
    "EMBED_MODEL_ENGLISH", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBED_MODEL_BANGLA = os.getenv(
    "EMBED_MODEL_BANGLA", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
MODEL_NAME = "google/mt5-small"
GROQ_MODEL = "llama-3.1-8b-instant"
PERSIST_DIRECTORY = "db_Policy"
COLLECTION_NAME = "Information_chunks"
RETRIEVAL_TOP_K = 5
MIN_RELEVANCE_SCORE = 0.2
HISTORY_MAX_TURNS = 8
REWRITE_HISTORY_TURNS = 6
STRICT_MIN_RELEVANCE_SCORE = 0.45
STRICT_MIN_RELEVANCE_SCORE_BANGLA = float(
    os.getenv("STRICT_MIN_RELEVANCE_SCORE_BANGLA", "0.20")
)
