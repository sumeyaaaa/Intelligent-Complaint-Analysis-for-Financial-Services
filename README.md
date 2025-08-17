

# 🧠 Intelligent Complaint Analysis Assistant

This project implements an advanced **Retrieval-Augmented Generation (RAG)** system designed to help internal teams at **CrediTrust Financial** analyze and understand customer complaints. The application features a multi-page Streamlit dashboard that allows for interactive data analysis and conversational Q\&A with an AI assistant grounded in real-world customer feedback from the Consumer Financial Protection Bureau (CFPB).


-----

## 🎯 Project Objective

To transform thousands of raw, unstructured complaint narratives into a strategic asset. The goal is to build an interactive AI assistant that empowers product, compliance, and support teams to:

  - Identify major complaint trends in minutes instead of days.
  - Answer product-specific questions with evidence-backed insights, without needing data analysts.
  - Proactively discover and address customer pain points before they escalate.

-----

## 🏛️ System Architecture

The project is built on a robust, modular architecture designed for performance and accuracy.

1.  **Data Processing Pipeline**: A memory-efficient pipeline processes the full dataset of over 400,000 complaints. It filters for relevant products, cleans the narratives, and uses a data-driven chunking strategy (`chunk_size=300`, `overlap=20`) to create focused, semantically meaningful text units.
2.  **Advanced RAG Core**: The "brain" of the system is a sophisticated `RAGPipeline` class that orchestrates several models:
      * **Embedding Model**: Uses `BAAI/bge-base-en-v1.5` to convert text chunks into high-quality vector embeddings.
      * **Vector Store**: A **FAISS** index stores approximately 1.9 million vectors for high-speed retrieval.
      * **Re-ranker Model**: A `CrossEncoder` model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) performs a crucial second-pass scoring of retrieved documents to ensure the highest possible relevance.
      * **Sentiment Model**: A `distilbert-base-uncased` model provides real-time sentiment analysis of the retrieved sources.
      * **Generative LLM**: `google/flan-t5-base` is used to synthesize the final, context-aware answer.
3.  **Comprehensive Evaluation Suite**: The pipeline's performance is measured using a rigorous, two-pronged approach:
      * **Manual Qualitative Scoring**: A human-in-the-loop process for scoring answer quality and relevance on a 1-5 scale.
      * **Automated Quantitative Metrics**: The **RAGAs** framework provides objective scores for metrics like `faithfulness`, `answer_relevancy`, and `context_precision`.
4.  **Multi-Page Streamlit UI**: A user-friendly web application provides several key functionalities:
      * **Homepage**: An overview of the project and key EDA metrics.
      * **Analysis Dashboard**: An interactive tool for filtering and visualizing complaint data, with an option to generate downloadable PDF reports.
      * **RAG Chatbot**: A conversational interface for asking questions and receiving AI-generated answers with transparent, citable sources.

-----

## 🛠️ Tech Stack

  - **Backend**: Python 3.10+, Pandas
  - **Core RAG Libraries**: `sentence-transformers`, `faiss-cpu`, `transformers`, `langchain`
  - **Evaluation**: `ragas`, `datasets`
  - **Frontend**: `streamlit`
  - **Key Models**:
      - Embedding: `BAAI/bge-base-en-v1.5`
      - Re-ranking: `cross-encoder/ms-marco-MiniLM-L-6-v2`
      - Generation: `google/flan-t5-base`
      - Sentiment: `distilbert-base-uncased-finetuned-sst-2-english`

-----

## 📁 Project Structure

```
rag_chatbot/
│
├── data/
│   ├── raw/
│   └── filtered_complaints.csv
│   └── balanced_sampled_complaints.csv
│   └── vector_store/
│       ├── index_bge_base_300_20.faiss
│       └── meta_bge_base_300_20.csv
│
├── evaluation/
│   ├── evaluation_dataset.csv
│   └── comprehensive_evaluation_results.csv
│
├── pages/
│   ├── 2_💬_Chatbot.py
│   └── 3_📈_Analysis_and_Report.py
│
├── src/
│   ├── preprocess_data.py
│   ├── create_balanced_sample.py
│   ├── rag_pipeline.py
│   └── prompts.py
│
├── 1_🏠_Homepage.py             # Main Streamlit entry point
├── config.yaml                  # Central configuration file
├── run_indexing.py              # Script to build the FAISS index
├── requirements.txt
└── README.md
```

-----

## 🚀 How to Run the Application

1.  **Setup Environment**:

    ```bash
    # Clone the repository
    git clone <your-repo-url>
    cd rag_chatbot

    # Create and activate a virtual environment
    python -m venv venv
    venv\Scripts\activate

    # Install dependencies
    pip install -r requirements.txt
    ```

2.  **Prepare Data and Build Index**:

    ```bash
    # (Optional) Run the preprocessing script if you have raw data
    # python src/preprocess_data.py

    # (Optional) Create the balanced sample dataset
    # python src/create_balanced_sample.py

    # Run the indexing script (this will take 1-2 hours on a CPU)
    python -u run_indexing.py
    ```

3.  **Launch the Streamlit App**:

    ```bash
    streamlit run 1_🏠_Homepage.py
    ```

-----

## 📊 Evaluation Results Example

The comprehensive evaluation combines manual scores with automated RAGAs metrics.

| Question | Generated Answer | Manual Score (1-5) | RAGAs Faithfulness | RAGAs Answer Relevancy |
| :--- | :--- | :---: | :---: | :---: |
| What issues are reported with credit card disputes? | Users report issues primarily related to billing disputes and unauthorized or fraudulent charges on their accounts. | 5 | 1.00 | 0.98 |
| Why do users complain about savings accounts? | Complaints often mention issues with fraud detection algorithms that lock accounts, and difficulties in accessing funds during investigations. | 4 | 1.00 | 0.95 |

-----

## 🔮 Future Work & Feature Plan

  - [ ] **Complaint Severity Classification**: Implement a text-classification model to automatically tag complaints by urgency (e.g., `Critical`, `High`, `Medium`), allowing teams to prioritize responses.
  - [ ] **Proactive Trend & Anomaly Detection**: Add a time-series analysis feature to the dashboard that automatically detects and highlights unusual spikes in complaint volume for specific products or issues.
  - [ ] **Automated Topic Modeling**: Integrate a library like **BERTopic** to generate weekly or monthly reports that automatically discover and summarize hidden, recurring complaint themes.