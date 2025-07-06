import sys
import os

# Add the src directory to the Python path
current_path = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_path, "..", "src")
src_path = os.path.abspath(src_path)

if src_path not in sys.path:
    sys.path.append(src_path)

print("✅ src path added:", src_path)

from embedding import EmbeddingModel
from chunking import chunk_with_langchain
from indexing import FaissIndexer
import pandas as pd
from tqdm import tqdm
import os

# --- Config ---
CSV_PATH = r"C:\Users\ABC\Desktop\10Acadamy\week_6\Intelligent-Complaint-Analysis-for-Financial-Services\data\clean\complaints_clean.csv"
INDEX_PATH = r"C:\Users\ABC\Desktop\10Acadamy\week_6\Intelligent-Complaint-Analysis-for-Financial-Services\data\vector_store\index_300_50.faiss"
META_PATH = r"C:\Users\ABC\Desktop\10Acadamy\week_6\Intelligent-Complaint-Analysis-for-Financial-Services\data\vector_store\meta_300_50.csv"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
CHUNKSIZE = 1000

# --- Setup ---
os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
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
        indexer.add(embeddings, chunk_meta)
        all_metadata.extend(chunk_meta)

# Save index and metadata after all batches
indexer.save(INDEX_PATH, META_PATH)
print("✅ Finished embedding and indexing the full dataset.")
