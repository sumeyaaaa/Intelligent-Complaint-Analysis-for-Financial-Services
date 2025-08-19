# rag_pipeline.py 

import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from typing import Dict, List, Tuple


# rag_pipeline.py

import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from typing import Dict, List, Tuple

class RAGPipeline:
    def __init__(self, config: Dict):
        self.config = config
        
        # Load embedding model from config
        model_name = self.config['embedding']['model_name']
        print(f"Loading embedding model: {model_name}")
        self.embedder = SentenceTransformer(model_name, device='cpu')
        
        # Load FAISS index and metadata from config
        print("Loading FAISS index and metadata...")
        
        # Use the correct key: 'output'
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



