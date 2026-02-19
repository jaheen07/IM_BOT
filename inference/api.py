import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException
from langchain_community.document_loaders import TextLoader
from pydantic import BaseModel

from config.constants import COLLECTION_NAME, PERSIST_DIRECTORY
from inference.pipeline import MenstrualHealthRAG
from splitter.text_splitter import get_text_splitter
from vector_store.embedder import get_embedder
from vector_store.store import build_vector_store

router = APIRouter()
rag_instance = MenstrualHealthRAG()


class VectorDBRequest(BaseModel):
    language: str  # "bangla" or "english"


class ChatRequest(BaseModel):
    user_id: str
    question: str


class ResetChatRequest(BaseModel):
    user_id: str


@router.post("/build-vectordb")
def build_vector_db(request: VectorDBRequest):
    lang = request.language.lower()
    if lang not in ["bangla", "english"]:
        raise HTTPException(status_code=400, detail="Language must be 'bangla' or 'english'.")

    data_dir = Path(f"data/raw/{lang}")
    if not data_dir.exists():
        raise HTTPException(status_code=404, detail=f"Directory {data_dir} not found.")

    files = list(data_dir.glob("*.txt"))
    if not files:
        raise HTTPException(status_code=404, detail=f"No .txt files found in {data_dir}.")

    splitter = get_text_splitter()
    embedder = get_embedder()

    documents = []
    for file in files:
        loader = TextLoader(str(file), encoding="utf-8")
        loaded_docs = loader.load()
        split_docs = splitter.split_documents(loaded_docs)
        for idx, doc in enumerate(split_docs):
            doc.metadata["source"] = file.name
            doc.metadata["language"] = lang
            doc.metadata["chunk_id"] = idx
        documents.extend(split_docs)

    if not documents:
        raise HTTPException(status_code=400, detail="No documents found after splitting.")

    persist_dir = f"{PERSIST_DIRECTORY}/{COLLECTION_NAME}_{lang}"
    if Path(persist_dir).exists():
        shutil.rmtree(persist_dir)

    build_vector_store(documents, embedder, persist_dir, collection_name=f"{lang}_chunks")
    rag_instance.invalidate_vectordb(lang)

    return {
        "message": f"Stored {len(documents)} chunks into ChromaDB at '{persist_dir}'",
        "chunks_stored": len(documents),
        "path": persist_dir
    }


@router.post("/chat")
def chat_endpoint(request: ChatRequest):
    query = request.question.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        response = rag_instance.get_response(query=query, user_id=request.user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")

    return response


@router.post("/chat/reset")
def reset_chat(request: ResetChatRequest):
    rag_instance.clear_history(request.user_id)
    return {"message": f"Chat history cleared for user_id '{request.user_id}'."}
