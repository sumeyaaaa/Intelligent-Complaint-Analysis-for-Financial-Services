import faiss
import pandas as pd

class FaissIndexer:
    def __init__(self, dim):
        self.index = faiss.IndexFlatIP(dim)
        self.metadata = []

    def add(self, vectors, meta_batch):
        self.index.add(vectors)
        self.metadata.extend(meta_batch)

    def save(self, index_path, meta_path):
        faiss.write_index(self.index, index_path)
        pd.DataFrame(self.metadata).to_csv(meta_path, index=False)
        print(f"\n✅ Saved FAISS index to {index_path} and metadata to {meta_path}")
