import pandas as pd
from datetime import datetime
import re
# Add this line at the top of your script
import sys 

def get_filtered_rag_data(df):
    """
    Simulates a data retrieval from a filtered data source.
    In a real RAG system, this would be your vector store.
    """
    # For this example, we'll just use the 'Consumer complaint narrative'
    # as our "knowledge base"
    return df['Consumer complaint narrative'].dropna().tolist()

def process_rag_query(prompt: str, df: pd.DataFrame) -> str:
    """
    Processes a user query by filtering the DataFrame based on the date
    and then providing a simulated answer.
    """
    # A simple, rule-based system to demonstrate context awareness
    prompt_lower = prompt.lower()
    
    # 1. Check for loan-related questions and filter
    if "loan" in prompt_lower or "mortgage" in prompt_lower or "credit" in prompt_lower:
        filtered_by_product = df[df['Product'].str.contains('Mortgage|Credit', case=False, na=False)]
        
        if not filtered_by_product.empty:
            # Get top companies for the filtered product
            top_companies = filtered_by_product['Company'].value_counts().head(3)
            response = (
                f"Based on the data from **{df['Date received'].min().date()} to {df['Date received'].max().date()}**, "
                f"I found {len(filtered_by_product):,} complaints related to loans and credit.\n\n"
                f"The top companies for these complaints are:\n\n"
                f"- **{top_companies.index[0]}**: {top_companies.iloc[0]} complaints\n"
                f"- **{top_companies.index[1]}**: {top_companies.iloc[1]} complaints\n"
                f"- **{top_companies.index[2]}**: {top_companies.iloc[2]} complaints"
            )
        else:
            response = "I couldn't find any loan or credit-related complaints in the selected date range."
    
    # 2. Check for general questions about total complaints
    elif "how many" in prompt_lower or "total complaints" in prompt_lower:
        total = len(df)
        response = f"There are a total of **{total:,}** complaints in the selected date range."
        
    # 3. Handle other simple questions or provide a default response
    else:
        response = (
            "I'm a simple bot. I can answer questions about the number of complaints, "
            "or specific loan-related problems within the date range. "
            "Please try a more specific question."
        )

    return response