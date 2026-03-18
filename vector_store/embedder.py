from langchain_community.embeddings import HuggingFaceEmbeddings

from config.constants import EMBED_MODEL_BANGLA, EMBED_MODEL_ENGLISH


def _normalize_language(language: str) -> str:
    lang = (language or "").lower().strip()
    return lang if lang in {"bangla", "english"} else "english"


def get_embed_model(language: str = "english") -> str:
    lang = _normalize_language(language)
    return EMBED_MODEL_BANGLA if lang == "bangla" else EMBED_MODEL_ENGLISH


def get_embedder(language: str = "english"):
    return HuggingFaceEmbeddings(model_name=get_embed_model(language))
