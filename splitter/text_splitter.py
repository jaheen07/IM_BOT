from langchain.text_splitter import RecursiveCharacterTextSplitter

from config.constants import CHUNK_OVERLAP, CHUNK_SIZE


def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
