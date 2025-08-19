import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
from typing import Dict, List, Tuple
import os

class RAGPipeline:
    def __init__(self):
        # The 'config' parameter is removed from the __init__ method signature
        print("--- Initializing RAG Pipeline ---")

        # Hardcoded paths, replacing the config dictionary
        # It's best to use os.path.join for cross-platform compatibility
        index_path = os.path.join("data", "vector_store11", "index_bge_base_300_20.faiss")
        meta_path = os.path.join("data", "vector_store11", "meta_bge_base_300_20.csv")
        
        print(f"Loading FAISS index from: {index_path}")
        self.index = faiss.read_index(index_path)
        
        self.metadata = pd.read_csv(meta_path, encoding='latin-1', engine='python')
        
        # Hardcoded model names and other configurations
        self.embedder = SentenceTransformer("BAAI/bge-base-en-v1.5", device='cpu')
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.llm = pipeline("text2text-generation", model="google/flan-t5-large")
        self.sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

        self.prompt_template = (
            "You are a financial analyst assistant. "
            "Use ONLY the following complaint excerpts to answer the question. "
            "If the answer is not in the context, say you don't have enough information.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )

        print("--- RAG Pipeline Initialized Successfully ---")

    def query(self, question: str) -> Tuple[str, List[Dict], str]:
        # The rest of your query method remains unchanged
        # ...
        
        # We need the values for k, but since we removed the config, we can hardcode them
        retrieval_k = 5
        rerank_k = 3
        
        question_embedding = self.embedder.encode([question], normalize_embeddings=True).astype(np.float32)
        _, indices = self.index.search(question_embedding, retrieval_k)
        initial_chunks = self.metadata.iloc[indices[0]].to_dict('records')

        if not initial_chunks:
            return "Could not retrieve any documents.", [], "N/A"

        initial_chunk_texts = [chunk['chunk_text'] for chunk in initial_chunks]
        pairs = [[question, chunk] for chunk in initial_chunk_texts]
        scores = self.reranker.predict(pairs)
        
        chunk_score_pairs = sorted(zip(initial_chunks, scores), key=lambda x: x[1], reverse=True)
        top_chunks = [pair[0] for pair in chunk_score_pairs[:rerank_k]]
        
        source_texts = [chunk['chunk_text'] for chunk in top_chunks]

        print("Analyzing sentiment of retrieved sources...")
        sentiments = self.sentiment_pipeline(source_texts)
        
        positive_count = sum(1 for s in sentiments if s['label'] == 'POSITIVE')
        negative_count = len(sentiments) - positive_count
        avg_score = np.mean([s['score'] for s in sentiments])
        sentiment_summary = f"Sentiment of Sources: {negative_count} NEGATIVE, {positive_count} POSITIVE (Avg. Confidence: {avg_score:.2f})"
        
        context = "\n\n".join(source_texts)
        prompt = self.prompt_template.format(context=context, question=question)
        
        response = self.llm(
            prompt,
            max_new_tokens=256
        )[0]["generated_text"]
        
        return response, top_chunks, sentiment_summary