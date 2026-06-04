import os
import sys
import asyncio
import subprocess
import streamlit as st

@st.cache_resource(show_spinner="Initializing browser environment...")
def install_browser():
    try:
        subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            capture_output=True,
            text=True
        )
    except Exception as e:
        pass

install_browser()

from src import (
    scrape_single_url,
    chunk_documents,
    build_vector_store,
    setup_rag_chain
)

st.set_page_config(
    page_title="Enterprise Web RAG Agent",
    layout="centered"
)

st.markdown("""
    <style>
        .reportview-container {
            background: #ffffff;
        }
        div.stButton > button:first-child {
            background-color: #0f172a;
            color: #ffffff;
            border-radius: 4px;
            border: none;
        }
        div.stButton > button:first-child:hover {
            background-color: #1e293b;
            color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Web RAG Analytics Platform")
st.markdown("Extract web documentation into a semantic knowledge base for localized analysis.")
st.markdown("---")

if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("Configuration Error: GROQ_API_KEY missing in Streamlit Secrets.")
    st.stop()

@st.cache_resource
def get_cached_rag_chain(build_id: int):
    """
    Caches the chain infrastructure in memory. 
    The build_id parameter forces a cache refresh only when a new database is compiled.
    """
    return setup_rag_chain()

if "build_id" not in st.session_state:
    st.session_state["build_id"] = 0
if "messages" not in st.session_state:
    st.session_state.messages = []

st.subheader("Knowledge Base Configuration")

with st.container():
    col1, col2 = st.columns([3, 1], vertical_alignment="bottom")
    
    with col1:
        target_url = st.text_input(
            "Target URL Source",
            placeholder="https://example.com/docs",
            label_visibility="visible"
        )
    
    with col2:
        build_btn = st.button("Build Database", use_container_width=True)

    if build_btn:
        if not target_url:
            st.error("Validation Error: Please enter a URL source.")
        elif not target_url.startswith("http"):
            st.error("Validation Error: URL protocol must start with http:// or https://")
        else:
            with st.spinner("Executing document scraping..."):
                try:
                    docs = asyncio.run(scrape_single_url(target_url))
                    if not docs:
                        st.error("Ingestion Failure: Content retrieval returned empty dataset.")
                        st.stop()
                except Exception as e:
                    st.error(f"Ingestion Failure: {str(e)}")
                    st.stop()

            with st.spinner("Processing structural text segmentation..."):
                try:
                    chunks = chunk_documents(docs)
                    if not chunks:
                        st.error("Segmentation Failure: Extracted content falls below structural threshold.")
                        st.stop()
                except Exception as e:
                    st.error(f"Segmentation Failure: {str(e)}")
                    st.stop()

            with st.spinner("Generating local vector space index..."):
                try:
                    build_vector_store(chunks)
                except Exception as e:
                    st.error(f"Indexing Failure: {str(e)}")
                    st.stop()

            st.session_state["build_id"] += 1
            st.session_state["messages"] = []
            st.session_state["active_source"] = target_url
            st.toast("Knowledge base compiled successfully.")

if st.session_state["build_id"] > 0:
    st.info(f"Active Index Source: {st.session_state['active_source']}")
    rag_chain = get_cached_rag_chain(st.session_state["build_id"])
else:
    st.warning("System Status: Awaiting valid data source initialization.")
    rag_chain = None

st.markdown("---")

st.subheader("Query Workspace")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Enter evaluation query..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if rag_chain is None:
            st.warning("Action Required: Please initialize a URL source above to populate the knowledge base index.")
        else:
            try:
                response = st.write_stream(rag_chain.stream(prompt))
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
            except Exception as e:
                st.error(f"Execution Error: {str(e)}")
