import re
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain_core.documents import Document


def detect_content_type(text: str) -> str:
    """
    Detect structured vs unstructured content.
    Real markdown table has separator row |---|---|
    """
    if re.search(r'\|[-:]+\|', text):
        return "table"
    elif re.search(r'```[\s\S]*?```', text):
        return "code"
    else:
        return "text"


def chunk_tables(doc: Document) -> list:
    """
    MarkdownHeaderSplitter for tables —
    keeps column headers attached to rows.
    Falls back to text chunking if parsing fails.
    """
    headers  = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers,
        strip_headers=False
    )
    try:
        chunks       = splitter.split_text(doc.page_content)
        rec_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )
        result = []
        for chunk in chunks:
            if len(chunk.page_content) > 1000:
                sub = rec_splitter.split_text(chunk.page_content)
                for s in sub:
                    result.append(Document(
                        page_content=s,
                        metadata={
                            **doc.metadata,
                            "content_type": "table"
                        }
                    ))
            else:
                result.append(Document(
                    page_content=chunk.page_content,
                    metadata={
                        **doc.metadata,
                        "content_type": "table"
                    }
                ))
        return result if result else chunk_text(doc)
    except Exception:
        return chunk_text(doc)


def chunk_text(doc: Document) -> list:
    """
    RecursiveCharacterTextSplitter for plain text.
    1000 char chunks (~200 words) for fast retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents([doc])
    for chunk in chunks:
        chunk.metadata["content_type"] = "text"
    return chunks


def chunk_documents(docs: list) -> list:
    """
    Smart chunking pipeline:
    - Tables → MarkdownHeaderSplitter (preserves structure)
    - Text   → RecursiveCharacterSplitter (~200 word blocks)
    - Filters out tiny chunks under 30 chars
    """
    all_chunks = []

    for doc in docs:
        content_type = detect_content_type(doc.page_content)
        if content_type == "table":
            chunks = chunk_tables(doc)
            print(f"📊 Table detected  → {len(chunks)} chunks")
        else:
            chunks = chunk_text(doc)
            print(f"📝 Text detected   → {len(chunks)} chunks")
        all_chunks.extend(chunks)

    before     = len(all_chunks)
    all_chunks = [
        c for c in all_chunks
        if len(c.page_content.strip()) >= 30
    ]
    filtered   = before - len(all_chunks)
    if filtered:
        print(f"🧹 Filtered {filtered} tiny chunks")

    print(f"✅ Total: {len(all_chunks)} chunks")
    return all_chunks
