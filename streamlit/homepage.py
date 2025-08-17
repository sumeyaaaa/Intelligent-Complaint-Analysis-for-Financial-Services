import streamlit as st
import pandas as pd
import os
import sys

# --- Set up project root and source path ---
# os.path.dirname(__file__) gets the directory of the current script (1_Homepage.py).
# We then go up one directory to get the project root.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

# --- Caching to load and analyze data only once ---
@st.cache_data
def load_and_analyze_data():
    try:
        # Construct the path more explicitly to avoid issues with '..'.
        # The correct path is project_root -> data -> clean -> complaints_clean.csv
        data_path = os.path.join(project_root, "data", "clean", "complaints_clean.csv")
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        st.error(f"Filtered data file not found at: {data_path}. Please ensure 'complaints_clean.csv' is in the correct directory.")
        return pd.DataFrame()

def calculate_eda_metrics(df):
    if df.empty:
        return {
            "Total Complaints": 0,
            "Complaints with Narrative": 0,
            "Percentage with Narrative": 0,
            "Median words / narrative": 0,
            "95th percentile word count": 0,
            "Max words / narrative": 0
        }
    
    # Calculate total number of complaints.
    total_complaints = len(df)
    
    # Calculate complaints with narratives and their word counts.
    narrative_column = "Consumer complaint narrative"
    narratives_with_content = df[df[narrative_column].notna() & (df[narrative_column].str.strip() != '')]
    total_with_narrative = len(narratives_with_content)
    
    # Calculate the percentage of complaints with narratives.
    percentage_with_narrative = (total_with_narrative / total_complaints) * 100 if total_complaints > 0 else 0
    
    # Calculate word count for each narrative.
    # The .str.len() method can be used to count the number of characters in a string.
    narratives_with_content['word_count'] = narratives_with_content[narrative_column].str.split().str.len()
    
    # Calculate EDA metrics.
    median_words = narratives_with_content['word_count'].median()
    p95_words = narratives_with_content['word_count'].quantile(0.95)
    max_words = narratives_with_content['word_count'].max()
    
    return {
        "Total Complaints": total_complaints,
        "Complaints with Narrative": total_with_narrative,
        "Percentage with Narrative": percentage_with_narrative,
        "Median words / narrative": median_words,
        "95th percentile word count": p95_words,
        "Max words / narrative": max_words
    }

# --- Main Streamlit App ---
st.set_page_config(page_title="RAG Finance Project", layout="centered")

st.title("💡 Intelligent Complaint Analysis Assistant")
st.markdown("""
*This RAG assistant lets CrediTrust teams explore real customer-complaint
narratives (CFPB public data) across five products.*
""")


# Load and compute metrics
df_complaints = load_and_analyze_data()
eda_metrics = calculate_eda_metrics(df_complaints)

# --- EDA Highlights (Dynamic Content) ---
st.subheader("📊 EDA highlights")
eda_data = {
    "Metric": ["Total Complaints", "Complaints with Narrative", "Percentage with Narrative", "Median words / narrative", "95th percentile word count", "Max words / narrative"],
    "Value": [
        f"{eda_metrics['Total Complaints']:,}",
        f"{eda_metrics['Complaints with Narrative']:,}",
        f"{eda_metrics['Percentage with Narrative']:.2f} %",
        f"{eda_metrics['Median words / narrative']:.0f}",
        f"{eda_metrics['95th percentile word count']:.0f}",
        f"{eda_metrics['Max words / narrative']:.0f}"
    ]
}
eda_df = pd.DataFrame(eda_data)
st.table(eda_df)

# --- Evaluation Snapshot (Static Content) ---
st.subheader("📝 Evaluation Snapshot")
st.warning("Note: This table is static. For dynamic evaluations, see the Chatbot page.")
eval_data = {
    "Question": ["BNPL pain-points", "Credit-card dispute issues", "Savings-account complaints", "Money-transfer problems"],
    "Score (1-5)": [3, 4, 2, 3]
}
eval_df = pd.DataFrame(eval_data)
st.table(eval_df)

st.markdown("""
*(Docs and code modules live in **rag_finance/src/**.
Chunk = 100 words, overlap = 20; embeddings = BAAI/bge-base-en-v1.5; index = FAISS Flat.)*
""")

st.info("👈 Navigate to the pages in the sidebar to get started!")