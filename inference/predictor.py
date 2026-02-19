from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os
from config.constants import GROQ_MODEL

load_dotenv()


def get_llm():
    return ChatOllama(
        model=os.getenv('OLLAMA_MODEL', 'llama3.1:8b'),
        base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
        temperature=0
    )

