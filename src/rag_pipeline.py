# In your main script, e.g., rag_pipeline.py or a notebook

import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from typing import Dict, List, Tuple

# --- Configuration for your OLD model and data ---
# This dictionary replaces hardcoded paths and model names.
config = {
    "embedding": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2"
    },
    "llm": {
        "model_name": "google/flan-t5-base",
        "max_new_tokens": 256
    },
    "retrieval": {
        "k": 5 # Number of chunks to retrieve
    },
    "prompt": {
        "template": (
            "You are a financial analyst assistant. "
            "Use ONLY the following complaint excerpts to answer the question. "
            "If the answer is not in the context, say you don't have enough information.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
    },
    "output": {
        "index_path": "data/vector_store/index_300_50.faiss",
        "meta_path": "data/vector_store/meta_300_50.csv"
    }
}

class RAGPipeline:
    """
    A refactored RAG pipeline that is driven by a configuration dictionary.
    This version performs direct FAISS retrieval without a re-ranker.
    """
    def __init__(self, config: Dict):
        self.config = config
        
        # Load embedding model from config
        model_name = self.config['embedding']['model_name']
        print(f"Loading embedding model: {model_name}")
        self.embedder = SentenceTransformer(model_name, device='cpu')
        
        # Load FAISS index and metadata from config
        print("Loading FAISS index and metadata...")
        self.index = faiss.read_index(self.config['output']['index_path'])
        self.metadata = pd.read_csv(self.config['output']['meta_path'])
        
        # Load LLM from config
        llm_name = self.config['llm']['model_name']
        print(f"Loading LLM: {llm_name}")
        self.llm = pipeline("text2text-generation", model=llm_name)

        # Load prompt template from config
        self.prompt_template = self.config['prompt']['template']

    def query(self, question: str) -> Tuple[str, List[Dict]]:
        """
        Executes the RAG pipeline: embed, retrieve, and generate.

        Args:
            question (str): The user's question.

        Returns:
            Tuple[str, List[Dict]]: The generated answer and a list of source chunks.
        """
        print("1. Embedding the question...")
        question_embedding = self.embedder.encode([question], normalize_embeddings=True).astype(np.float32)
        
        print("2. Retrieving top chunks from FAISS...")
        retrieval_k = self.config['retrieval']['k']
        _, indices = self.index.search(question_embedding, retrieval_k)
        
        # Get the full dictionary for each retrieved chunk
        top_chunks = self.metadata.iloc[indices[0]].to_dict('records')
        
        # Extract just the text for the context
        source_texts = [chunk['chunk_text'] for chunk in top_chunks]
        context = "\n\n".join(source_texts)

        print("3. Generating the final answer...")
        prompt = self.prompt_template.format(context=context, question=question)
        
        response = self.llm(
            prompt,
            max_new_tokens=self.config['llm']['max_new_tokens']
        )[0]["generated_text"]
        
        return response, top_chunks

# --- Example of how to use the class ---

