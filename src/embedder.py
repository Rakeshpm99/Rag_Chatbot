from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma  

CHROMA_PATH = "./chroma_db"

def get_embedding_engine():
    return HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1.5",
        model_kwargs={'trust_remote_code': True}
    )

def populate_vector_store(chunks: List[Document]) -> Chroma:
    embeddings = get_embedding_engine()
    vector_store = Chroma(embedding_function=embeddings, persist_directory=CHROMA_PATH)
    if chunks:
        vector_store.add_documents(chunks)
    return vector_store
