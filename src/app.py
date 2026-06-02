import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.embedder import build_vector_store

def run_basic_chat(query, chunks):
    db = build_vector_store(chunks)
    retriever = db.as_retriever()
    llm = ChatGroq(model="llama-3.1-8b-instant")
    prompt = ChatPromptTemplate.from_template("Context: {context}\n\nQuestion: {question}")
    
    chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    return chain.invoke(query)
