from langchain_community.embeddings import HuggingFaceEmbeddings

from config.constants import EMBED_MODEL


def get_embedder(model_name: str = EMBED_MODEL):
    return HuggingFaceEmbeddings(model_name=model_name)
