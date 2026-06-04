import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ── SQLite LLM Cache ───────────────────────────────────────
# Same question = instant reply, no Groq API call
set_llm_cache(SQLiteCache(database_path=".langchain.db"))


def format_docs(docs):
    return "\n\n---\n\n".join([
        f"[Source: {d.metadata.get('source', 'Unknown')}]\n{d.page_content}"
        for d in docs
    ])


def setup_rag_chain():
    """
    LCEL RAG chain:
    - streaming=True  → words appear as generated
    - SQLiteCache     → repeat questions instant
    - llama-3.1-8b-instant → lowest latency model
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )
    db         = Chroma(
        persist_directory="./chroma_db_test",
        embedding_function=embeddings
    )
    retriever  = db.as_retriever(search_kwargs={"k": 5})
    llm        = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        streaming=True
    )
    prompt     = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a precise technical assistant. "
            "Answer the user's question based strictly "
            "on the provided context scraped from websites. "
            "If the answer is not in the context, say clearly: "
            "'I lack sufficient context to answer this.'\n\n"
            "Context:\n{context}"
        )),
        ("human", "{question}")
    ])
    return (
        {
            "context":  retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
