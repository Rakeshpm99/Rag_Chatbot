from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def build_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5", model_kwargs={'trust_remote_code': True})
    return Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
