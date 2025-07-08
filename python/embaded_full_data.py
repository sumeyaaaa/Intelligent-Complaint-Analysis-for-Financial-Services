import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

# --- Add `src` to path ---
current_path = Path(__file__).resolve().parent
src_path = current_path.parent / "src"
sys.path.append(str(src_path))

print("✅ src path added:", src_path)

# --- Import project modules ---
from embedding import EmbeddingModel
from chunking import chunk_with_langchain
from indexing import FaissIndexer

# --- Config ---
CSV_PATH = r"C:\Users\ABC\Desktop\10Acadamy\week_6\Intelligent-Complaint-Analysis-for-Financial-Services\data\clean\complaints_clean.csv"
INDEX_PATH = r"C:\Users\ABC\Desktop\10Acadamy\week_6\Intelligent-Complaint-Analysis-for-Financial-Services\data\vector_store\index_300_50.faiss"
META_PATH = r"C:\Users\ABC\Desktop\10Acadamy\week_6\Intelligent-Complaint-Analysis-for-Financial-Services\data\vector_store\meta_300_50.csv"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
CHUNKSIZE = 1000

# --- Setup ---
os.makedirs(Path(INDEX_PATH).parent, exist_ok=True)
embedder = EmbeddingModel()
indexer = FaissIndexer(dim=embedder.get_dimension())
all_metadata = []

# --- Stream and process the full file in chunks ---
reader = pd.read_csv(CSV_PATH, chunksize=CHUNKSIZE)

for df in tqdm(reader, desc="Processing complaints"):
    chunk_texts = []
    chunk_meta = []

    for _, row in df.iterrows():
        complaint_id = row.get("Complaint ID")
        product = row.get("Product")
        text = row.get("Consumer complaint narrative", "")

        if not isinstance(text, str) or not text.strip():
            continue

        chunks = chunk_with_langchain(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        for chunk in chunks:
            chunk_texts.append(chunk)
            chunk_meta.append({
                "complaint_id": complaint_id,
                "product": product,
                "chunk_text": chunk
            })

    if chunk_texts:
        embeddings = embedder.encode(chunk_texts)
        embeddings = np.array(embeddings).astype("float32")  # 🔧 Required for FAISS
        indexer.add(embeddings, chunk_meta)
        all_metadata.extend(chunk_meta)

# --- Final save ---
indexer.save(INDEX_PATH, META_PATH)
print("✅ Finished embedding and indexing the full dataset.")
