# run_indexing.py (Updated Version with Sampling + Adaptive Batch)

import os
import yaml
import pandas as pd
import faiss
import numpy as np
import psutil   # <-- for checking available memory
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_with_langchain(text, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_safe_batch_size(base_batch=1000):
    """Dynamically scale batch size depending on free RAM"""
    free_gb = psutil.virtual_memory().available / (1024**3)
    if free_gb > 16:
        return base_batch * 3   # use 3000 if plenty of RAM
    elif free_gb > 8:
        return base_batch * 2   # 2000 if mid RAM
    else:
        return base_batch       # 1000 for low memory machines

def main():
    """Main function to run the embedding and indexing pipeline."""
    
    # 1. Load Configuration
    print("Loading configuration...")
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # 2. Initialize the Embedding Model
    model_name = config['embedding']['model_name']
    print(f"Initializing embedding model: {model_name}")
    embedder = SentenceTransformer(model_name)
    
    # 3. Create FAISS index
    embedding_dim = embedder.get_sentence_embedding_dimension()
    print(f"Creating new FAISS index with dimension: {embedding_dim}")
    index = faiss.IndexFlatL2(embedding_dim)
    metadata_list = []

    # 4. Load full CSV and sample 10k per Product
    # CORRECTED LINE: Access the path using a standard key, e.g., 'csv_path'
    csv_path = config['data']['csv_path']
    print(f"Loading and sampling data from {csv_path}...")
    df = pd.read_csv(csv_path, dtype=str)

    df.dropna(subset=['Consumer complaint narrative', 'Product'], inplace=True)
    
    sampled_df = df.groupby("Product").apply(
        lambda x: x.sample(n=min(10000, len(x)), random_state=42)
    ).reset_index(drop=True)

    print(f"Sampled dataset size: {len(sampled_df)} rows")

    # 5. Process sampled data in adaptive batches
    base_batch = config['processing']['batch_size']
    chunk_size = config['processing']['chunk_size']
    chunk_overlap = config['processing']['chunk_overlap']
    safe_batch_size = get_safe_batch_size(base_batch)

    print(f"Processing {len(sampled_df)} records in batches of {safe_batch_size}...")

    for start in tqdm(range(0, len(sampled_df), safe_batch_size), desc="Processing Batches"):
        df_batch = sampled_df.iloc[start:start+safe_batch_size]

        # Chunking
        df_batch['chunks'] = df_batch['Consumer complaint narrative'].apply(
            lambda text: chunk_with_langchain(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        )
        df_exploded = df_batch.explode('chunks').rename(columns={'chunks': 'chunk_text'}).dropna(subset=['chunk_text'])
        
        chunk_texts = df_exploded['chunk_text'].tolist()
        if not chunk_texts:
            continue

        # Embedding
        embeddings = embedder.encode(chunk_texts, normalize_embeddings=True, show_progress_bar=True)
        chunk_meta = df_exploded[["Complaint ID", "Product", "Date received", "Issue", "chunk_text"]].to_dict('records')

        index.add(np.array(embeddings, dtype=np.float32))
        metadata_list.extend(chunk_meta)

    # 6. Save results
    index_path = config['data']['index_path']
    meta_path = config['data']['meta_path']
    output_dir = os.path.dirname(index_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\nSaving FAISS index to {index_path}...")
    faiss.write_index(index, index_path)
    
    print(f"Saving metadata to {meta_path}...")
    pd.DataFrame(metadata_list).to_csv(meta_path, index=False)
    
    print(f"\n✅ Indexing complete.")

if __name__ == "__main__":
    main()