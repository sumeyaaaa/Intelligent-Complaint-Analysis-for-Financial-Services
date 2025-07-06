import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

class RAGPipeline:
    def __init__(self,
                 index_path=r"C:\Users\ABC\Desktop\10Acadamy\week_6\Intelligent-Complaint-Analysis-for-Financial-Services\data\clean\vector_store\index_300_50.faiss",
                 metadata_path=r"C:\Users\ABC\Desktop\10Acadamy\week_6\Intelligent-Complaint-Analysis-for-Financial-Services\data\clean\vector_store\meta_300_50.csv",
                 embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                 llm_model="google/flan-t5-base",
                 top_k=5):

        # Load FAISS index and metadata
        self.index = faiss.read_index(index_path)
        self.metadata = pd.read_csv(metadata_path)

        # Load embedder and normalize helper
        self.embedder = SentenceTransformer(embedding_model)
        self.dim = 384
        self.top_k = top_k

        # Load lightweight local LLM
        self.llm = pipeline("text2text-generation", model=llm_model)

    def _normalize(self, vecs):
        faiss.normalize_L2(vecs)
        return vecs
    def retrieve_chunks(self, query):
        vec = self.embedder.encode([query]).astype("float32")
        vec = self._normalize(vec)
        D, I = self.index.search(vec, self.top_k)

        chunks = self.metadata.loc[I[0], "chunk_text"].astype(str).tolist()
        return chunks


    def generate_answer(self, query, max_tokens=1200):
        chunks = self.retrieve_chunks(query)
        context = "\n\n".join(chunks)

    # Truncate context length to ~1200 tokens (adjust as needed)
        if len(context) > max_tokens * 4:  # rough estimate: 1 token ≈ 4 chars
           context = context[:max_tokens * 4]

        prompt = (
          "You are a financial analyst assistant for CrediTrust. "
           "Use ONLY the following complaint excerpts to answer the question. "
           "If the answer is not in the context, say you don't have enough information.\n\n"
           f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
           )

        response = self.llm(prompt, max_length=256)[0]["generated_text"]
        return response, chunks







       
