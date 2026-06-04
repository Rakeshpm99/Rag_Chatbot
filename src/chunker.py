from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks")
    return chunks
