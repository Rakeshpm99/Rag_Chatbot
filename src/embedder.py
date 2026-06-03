import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

def build_vector_store(chunks, persist_directory="./chroma_db"):
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    client = chromadb.PersistentClient(path=persist_directory)

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        collection_name="rag_collection"
    )
