import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

set_llm_cache(SQLiteCache(database_path=".langchain.db"))

def format_docs(docs):
    return "\n\n---\n\n".join([f"[Source: {d.metadata.get('source', 'Unknown')}]\n{d.page_content}" for d in docs])

def setup_rag_chain():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    db = Chroma(persist_directory="./chroma_db_test", embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 5})

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer strictly based on the context. If unknown, say 'I lack sufficient context.'\n\nContext:\n{context}"),
        ("human", "{question}")
    ])

    return ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
