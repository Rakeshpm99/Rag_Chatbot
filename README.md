# RAG-Powered Website Chatbot

A high-performance,  Retrieval-Augmented Generation (RAG) chatbot designed to  scrapes any website and answers questions with minimal latency.

## 🔗 Live Demo
Experience the application live here: [https://rag-web-chatbot.streamlit.app](https://rag-web-chatbot.streamlit.app)

## 📌 Problem Statement
Build a chatbot that can ingest any given URL, scrape relevant
content, and use RAG to answer user questions accurately based
on the collected content — with minimal latency and robust
handling of structured and unstructured data.

## 🏗️ Architecture
URL Input → Crawl4AI Scraper → Smart Chunker → BAAI Embeddings → ChromaDB → LLM → Answer


## ⚙️ Key Technical Features
**⚡ Minimum Latency**
- SQLiteCache — repeat questions answered instantly, zero API calls
- Added Streaming on ChatGroq — words appear as generated
- Used lightweight emdedding model, loads once, stays cached
- `word_count_threshold=10` discards empty pages and boilerplate, ensuring only dense, high-value content reaches the vector index.
- `magic=True` on Crawl4AI — instant clean markdown extraction
  
**📊 Structured & Unstructured Data**
- Auto-detects content type using `re.search(r'\|[-:]+\|', text)`
- Tables → `MarkdownHeaderTextSplitter` keeps headers attached to rows
- Plain text → `RecursiveCharacterTextSplitter` with 1000 char chunks
- 150 char overlap prevents context loss at chunk boundaries
- Chunks under 30 chars filtered out automatically
- Falls back to recursive splitting if markdown parsing fails


## 🛠️ Tech Stack
* **Scraping**: Crawl4ai
* **Embeddings**: HuggingFace (Local)
* **Vector Database**: ChromaDB
* **Orchestration**: LangChain
* **UI**: Streamlit


##  🚀 Setup & Usage
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Set your Groq API key : `GROQ_API_KEY="your_key_here"` 
4. Run the application: `streamlit run app.py`.

---
