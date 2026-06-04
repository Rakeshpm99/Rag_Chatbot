from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def build_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="./chroma_db_test"
    )
