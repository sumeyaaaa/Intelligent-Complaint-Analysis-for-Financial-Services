import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textwrap

def get_rag_response(prompt: str, df: pd.DataFrame) -> str:
    """
    Retrieves a response for a user query using a simple RAG model.
    The response is generated from the most relevant complaint narratives
    found in the filtered DataFrame.
    """
    # Use only the 'Consumer complaint narrative' column for RAG
    narratives = df['Consumer complaint narrative'].dropna().tolist()

    if not narratives:
        return "Sorry, I can't find any relevant narratives in the selected date range to answer your question."

    # Step 1: Retrieval (Find relevant narratives)
    # Using TF-IDF for a lightweight similarity search
    vectorizer = TfidfVectorizer().fit(narratives)
    narrative_vectors = vectorizer.transform(narratives)
    prompt_vector = vectorizer.transform([prompt])

    # Compute cosine similarity between the prompt and all narratives
    similarities = cosine_similarity(prompt_vector, narrative_vectors)
    most_similar_indices = similarities.argsort()[0][-3:][::-1] # Get top 3 indices

    retrieved_narratives = [narratives[i] for i in most_similar_indices]
    
    # Step 2: Augmentation and Generation
    # Construct a response by combining the prompt with the retrieved narratives.
    # A real LLM would take this context and generate a fluid answer.
    # For this simulated model, we present the findings directly.
    response = (
        "Based on the most relevant complaints in the selected date range, here's what I found:\n\n"
        "**Relevant Complaint Narratives:**\n"
    )

    for i, narrative in enumerate(retrieved_narratives):
        wrapped_text = textwrap.fill(narrative, width=100)
        response += f"**- Narrative {i+1}:** {wrapped_text}\n\n"
    
    return response