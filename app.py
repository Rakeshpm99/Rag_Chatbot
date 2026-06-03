import os
import asyncio
import subprocess
import streamlit as st

# 1. Safer Cloud Deployment Setup
@st.cache_resource(show_spinner="Booting browser environment for cloud deployment...")
def install_browser():
    try:
        subprocess.run(["python", "-m", "playwright", "install", "chromium"], check=True)
    except Exception as e:
        st.error(f"Failed to install Chromium: {e}")

install_browser()

from src import scrape_single_url, chunk_documents, build_vector_store, setup_rag_chain

st.set_page_config(page_title="Web RAG Agent", page_icon="🤖")
st.title("🤖 Web RAG Agent")
st.markdown("Scrape any website and chat with its content securely using RAG.")

if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("❌ GROQ_API_KEY missing in Streamlit Advanced Settings -> Secrets!")
    st.stop()

with st.sidebar:
    st.header("⚙️ Ingestion Setup")
    target_url = st.text_input("Target URL", placeholder="https://example.com")
    
    if st.button("Build Knowledge Base"):
        if not target_url:
            st.error("Please provide a valid URL.")
        else:
            with st.spinner("Scraping DOM and building vector database..."):
                try:
                    docs = asyncio.run(scrape_single_url(target_url))
                    chunks = chunk_documents(docs)
                    
                    if not chunks:
                         st.error("❌ Text was found, but it was too short or improperly formatted to chunk.")
                    else:
                        build_vector_store(chunks)
                        st.session_state["rag_chain"] = setup_rag_chain()
                        st.success("🤖 Knowledge Base Ready!")
                        
                except Exception as e:
                    # THIS WILL PRINT OUR RAW ERROR
                    st.error(f"🚨 Diagnostic Error: {str(e)}")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about the website..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        if "rag_chain" in st.session_state:
            with st.spinner("Thinking..."):
                response = st.session_state["rag_chain"].invoke(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.warning("Please enter a URL and click 'Build Knowledge Base' in the sidebar first.")
