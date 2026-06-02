import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from src.embedder import build_vector_store

# Added caching
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

def run_basic_chat(query, chunks):
    db = build_vector_store(chunks)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict technical assistant. Answer ONLY based on the context below. If the answer is missing, say 'I lack sufficient context.'\n\nContext:\n{context}"),
        ("human", "{question}")
    ])
    
    chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    return chain.invoke(query)
