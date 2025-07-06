from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

class EmbeddingModel:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dim = 384  # Fixed dimension for MiniLM-L6-v2

    def encode(self, texts, batch_size=64, normalize=True):
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype=np.float32)
        if normalize:
            faiss.normalize_L2(embeddings)
        return embeddings

    def get_dimension(self):
        return self.dim