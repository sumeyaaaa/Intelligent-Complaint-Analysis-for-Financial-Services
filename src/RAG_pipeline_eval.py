# src\RAG_pipeline_eval.py
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
from typing import Dict, List, Tuple

class RAGPipeline:
    def __init__(self, config: Dict):
        self.config = config
        print("--- Initializing RAG Pipeline ---")
        self.embedder = SentenceTransformer(self.config['embedding']['model_name'], device='cpu')
        self.index = faiss.read_index(self.config['data']['index_path'])
        self.metadata = pd.read_csv(self.config['data']['meta_path'])
        self.reranker = CrossEncoder(self.config['reranker']['model_name'])
        self.llm = pipeline("text2text-generation", model=self.config['llm']['model_name'])
        
        # --- NEW: Load the Sentiment Analysis Model ---
        print("Loading Sentiment Analysis model...")
        self.sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        
        self.prompt_template = self.config['prompt']['template']
        print("--- RAG Pipeline Initialized Successfully ---")

    def query(self, question: str) -> Tuple[str, List[Dict], str]:
        # The return type is updated to include a string for the sentiment summary
        
        question_embedding = self.embedder.encode([question], normalize_embeddings=True).astype(np.float32)
        retrieval_k = self.config['retrieval']['k']
        _, indices = self.index.search(question_embedding, retrieval_k)
        initial_chunks = self.metadata.iloc[indices[0]].to_dict('records')
        
        if not initial_chunks:
            return "Could not retrieve any documents.", [], "N/A"

        rerank_k = self.config['reranker']['k']
        initial_chunk_texts = [chunk['chunk_text'] for chunk in initial_chunks]
        pairs = [[question, chunk] for chunk in initial_chunk_texts]
        scores = self.reranker.predict(pairs)
        
        chunk_score_pairs = sorted(zip(initial_chunks, scores), key=lambda x: x[1], reverse=True)
        top_chunks = [pair[0] for pair in chunk_score_pairs[:rerank_k]]
        
        source_texts = [chunk['chunk_text'] for chunk in top_chunks]

        # --- NEW: Perform and Summarize Sentiment Analysis ---
        print("Analyzing sentiment of retrieved sources...")
        sentiments = self.sentiment_pipeline(source_texts)
        
        # Aggregate the results into a simple summary
        positive_count = sum(1 for s in sentiments if s['label'] == 'POSITIVE')
        negative_count = len(sentiments) - positive_count
        avg_score = np.mean([s['score'] for s in sentiments])
        sentiment_summary = f"Sentiment of Sources: {negative_count} NEGATIVE, {positive_count} POSITIVE (Avg. Confidence: {avg_score:.2f})"
        
        # --- End of New Logic ---
        
        context = "\n\n".join(source_texts)
        prompt = self.prompt_template.format(context=context, question=question)
        
        response = self.llm(
            prompt,
            max_new_tokens=self.config['llm']['max_new_tokens']
        )[0]["generated_text"]
        
        # Update the return value to include the sentiment summary
        return response, top_chunks, sentiment_summary