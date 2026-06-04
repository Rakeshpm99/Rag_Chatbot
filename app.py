import os
import sys
import asyncio
import subprocess
import nest_asyncio
import streamlit as st

nest_asyncio.apply()


# ── Browser setup for cloud deployment ────────────────────
@st.cache_resource(show_spinner="Setting up browser environment...")
def install_browser():
    try:
        result = subprocess.run(
            [sys.executable, "-m", "playwright",
             "install", "chromium"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            st.error("🚨 Playwright install failed!")
            st.error(f"STDOUT: {result.stdout}")
            st.error(f"STDERR: {result.stderr}")
    except Exception as e:
        st.error(f"Subprocess error: {e}")

install_browser()

from src import (
    scrape_single_url,
    chunk_documents,
    build_vector_store,
    setup_rag_chain
)

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Web RAG Agent",
    page_icon="🤖"
)
st.title("🤖 Web RAG Agent")
st.markdown(
    "Scrape any website and chat with its content using RAG."
)

# ── Groq API key ───────────────────────────────────────────
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("❌ GROQ_API_KEY missing in Streamlit Secrets!")
    st.stop()

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Knowledge Base Setup")
    target_url = st.text_input(
        "Target URL",
        placeholder="https://example.com"
    )

    if st.button("Build Knowledge Base 🚀"):
        if not target_url:
            st.error("Please enter a valid URL.")
        elif not target_url.startswith("http"):
            st.error("URL must start with http:// or https://")
        else:
            # Step 1 — Scrape
            with st.spinner("🌐 Scraping website..."):
                try:
                    loop = asyncio.get_event_loop()
                    docs = loop.run_until_complete(
                        scrape_single_url(target_url)
                    )
                    if not docs:
                        st.error("❌ No content found.")
                        st.stop()
                    st.info(f"📄 Scraped {len(docs)} page(s)")
                except Exception as e:
                    st.error(f"🚨 Scrape error: {str(e)}")
                    st.stop()

            # Step 2 — Chunk
            with st.spinner("✂️ Chunking content..."):
                try:
                    chunks = chunk_documents(docs)
                    if not chunks:
                        st.error("❌ Content too short.")
                        st.stop()
                    st.info(f"✂️ {len(chunks)} chunks created")
                except Exception as e:
                    st.error(f"🚨 Chunk error: {str(e)}")
                    st.stop()

            # Step 3 — Embed + store
            with st.spinner("💾 Building vector store..."):
                try:
                    build_vector_store(chunks)
                    st.info("💾 Vector store ready!")
                except Exception as e:
                    st.error(f"🚨 Vector store error: {str(e)}")
                    st.stop()

            # Step 4 — RAG chain
            with st.spinner("🔗 Setting up RAG chain..."):
                try:
                    st.session_state["rag_chain"] = setup_rag_chain()
                    st.session_state["messages"]  = []
                    st.success(
                        f"✅ Ready! {len(chunks)} chunks indexed."
                    )
                except Exception as e:
                    st.error(f"🚨 Chain error: {str(e)}")
                    st.stop()

    # Status
    if "rag_chain" in st.session_state:
        st.success("✅ Knowledge base active")
    else:
        st.info("⬆️ Enter a URL to get started")

# ── Chat ───────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask anything about the website..."):
    st.session_state.messages.append({
        "role": "user", "content": prompt
    })
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if "rag_chain" not in st.session_state:
            st.warning("⚠️ Please load a URL first.")
        else:
            try:
                # Streaming — words appear as generated
                response = st.write_stream(
                    st.session_state["rag_chain"].stream(prompt)
                )
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": response
                })
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
