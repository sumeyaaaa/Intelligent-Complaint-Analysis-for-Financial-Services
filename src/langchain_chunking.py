from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_with_langchain(text, chunk_size=300, chunk_overlap=50):
    if not isinstance(text, str) or not text.strip():
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return [doc.page_content for doc in splitter.create_documents([text])]

def batch_chunk_texts(texts, chunk_size=300, chunk_overlap=50):
    all_chunks = []
    counts = []
    for text in texts:
        chunks = chunk_with_langchain(text, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)
        counts.append(len(chunks))
    return all_chunks, counts
