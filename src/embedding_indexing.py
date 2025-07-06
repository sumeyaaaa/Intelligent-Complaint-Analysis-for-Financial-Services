# src/embed_and_index.py

import os
import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Config ---
CSV_PATH = "data/filtered_complaints.csv"
INDEX_PATH = "vector_store/index_300_50.faiss"
META_PATH = "vector_store/meta_300_50.csv"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
CHUNKSIZE = 1000

# --- Setup ---
os.makedirs("vector_store", exist_ok=True)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
index = faiss.IndexFlatIP(384)
metadata = []

# --- Normalize helper ---
def normalize(vecs):
    faiss.normalize_L2(vecs)
    return vecs

# --- Stream and embed ---
reader = pd.read_csv(CSV_PATH, chunksize=CHUNKSIZE)
for df in tqdm(reader, desc="Embedding complaints"):
    chunk_texts, chunk_meta = [], []

    for _, row in df.iterrows():
        complaint_id = row.get("Complaint ID")
        product = row.get("Product")
        narrative = row.get("Consumer complaint narrative", "")

        if not isinstance(narrative, str) or not narrative.strip():
            continue

        docs = splitter.create_documents([narrative])
        for doc in docs:
            chunk = doc.page_content
            chunk_texts.append(chunk)
            chunk_meta.append({
                "complaint_id": complaint_id,
                "product": product,
                "chunk_text": chunk
            })

    if chunk_texts:
        vecs = embedder.encode(chunk_texts, batch_size=64)
        vecs = normalize(np.array(vecs).astype("float32"))
        index.add(vecs)
        metadata.extend(chunk_meta)

# --- Save outputs ---
faiss.write_index(index, INDEX_PATH)
pd.DataFrame(metadata).to_csv(META_PATH, index=False)
print("✅ FAISS index and metadata saved.")
