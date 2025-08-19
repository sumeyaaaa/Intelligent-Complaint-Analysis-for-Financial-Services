# run_indexing.py (Edited for a 20k-per-product balanced sample)

import os
import yaml
import pandas as pd
import faiss
import numpy as np
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

def main():
    """Main function to build a large, balanced sample index."""
    
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

    # 4. Load full CSV and create a balanced sample
    csv_path = config['data']['csv_path']
    print(f"Loading and sampling data from {csv_path}...")
    df = pd.read_csv(csv_path, dtype=str)
    df.dropna(subset=['Consumer complaint narrative', 'Product'], inplace=True)
    
    # --- KEY CHANGE: Sample size per product changed to 20000 ---
    SAMPLES_PER_PRODUCT = 20000
    
    sampled_df = df.groupby("Product").apply(
        lambda x: x.sample(n=min(SAMPLES_PER_PRODUCT, len(x)), random_state=42)
    ).reset_index(drop=True)

    print(f"Created a balanced sample with {len(sampled_df)} total records.")

    # 5. Process the sample dataset in batches
    print("Processing the sample dataset...")
    batch_size = config['processing']['batch_size']
    chunk_size = config['processing']['chunk_size']
    chunk_overlap = config['processing']['chunk_overlap']

    for start in tqdm(range(0, len(sampled_df), batch_size), desc="Processing Batches"):
        df_batch = sampled_df.iloc[start:start+batch_size]

        # Chunking
        df_batch['chunks'] = df_batch['Consumer complaint narrative'].apply(
            lambda text: chunk_with_langchain(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        )
        df_exploded = df_batch.explode('chunks').rename(columns={'chunks': 'chunk_text'}).dropna(subset=['chunk_text'])
        
        chunk_texts = df_exploded['chunk_text'].tolist()
        if not chunk_texts:
            continue

        # Embedding and Indexing
        embeddings = embedder.encode(chunk_texts, normalize_embeddings=True)
        chunk_meta = df_exploded[["Complaint ID", "Product", "Date received", "Issue", "chunk_text"]].to_dict('records')
        
        index.add(np.array(embeddings, dtype=np.float32))
        metadata_list.extend(chunk_meta)

    # 6. Save the index and metadata
    index_path = config['output']['index_path']
    meta_path = config['output']['meta_path']
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