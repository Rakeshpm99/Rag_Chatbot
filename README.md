# RAG Chatbot: Intelligent Web-RAG Pipeline

A high-performance, private Retrieval-Augmented Generation (RAG) chatbot designed to ingest, process, and query complex web content with minimal latency.

## 🔗 Live Demo
Experience the application live here: [https://rag-web-chatbot.streamlit.app](https://rag-web-chatbot.streamlit.app)

## 🚀 Overview
This project addresses the challenge of extracting and synthesizing information from large, unstructured web data. By utilizing a localized vector space and optimized chunking strategies, this chatbot provides fast, accurate answers while maintaining complete data privacy and high performance.

## 🏗️ Architecture
The system utilizes a modular, high-efficiency pipeline designed for precision:
1. **Web Scraping**: Efficient HTML extraction and noise filtering.
2. **Intelligent Chunking**: Document-aware splitting using **Markdown Header Splitters** for structure and **Recursive Character Splitters** for prose.
3. **Embedding & Vector Storage**: High-speed, local vectorization via ChromaDB.
4. **LLM Synthesis**: Streaming inference for real-time, low-latency responses.



## ⚙️ Key Technical Features
* **Minimal Latency Architecture**: Achieved via `@st.cache_resource` for state persistence and localized, in-memory vector storage, eliminating cloud bottlenecks.
* **Robust Data Handling**: Preserves structural integrity of complex elements like tables through layout-aware Markdown splitting, preventing the data fragmentation common in standard RAG pipelines.
* **Asynchronous Ingestion**: Non-blocking scraping ensures the system remains responsive during data indexing.
* **Streaming Inference**: Near-instant "Time-To-First-Token" for a fluid, expert-level user experience.

## 🛠️ Tech Stack
* **Framework**: Streamlit
* **Orchestration**: LangChain
* **Vector Database**: ChromaDB
* **Scraping**: Playwright / Async-enabled utilities
* **Embeddings**: HuggingFace (Local)

## 📦 Usage
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the application: `streamlit run app.py`.
4. Input a URL, build the knowledge base, and start querying!

---
*Engineered for real-world reliability and precise information synthesis.*
