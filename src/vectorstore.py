from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_PATH = "./chroma_db"

embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5")
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH
)

def store_chunks(chunks: list):
    vector_store.add_documents(chunks)
    print(f"Stored {len(chunks)} chunks")

def get_retriever(k: int = 5):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
