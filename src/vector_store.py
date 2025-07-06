import faiss
import numpy as np
import pickle
import os

def build_faiss_index(embeddings, metadata_list, save_path="vector_store/faiss_index"):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    faiss.write_index(index, f"{save_path}.index")

    with open(f"{save_path}_metadata.pkl", "wb") as f:
        pickle.dump(metadata_list, f)
