
# 🧠 Intelligent Complaint Analysis Assistant

An advanced Retrieval-Augmented Generation (RAG) system designed to help internal teams at CrediTrust Financial analyze and understand thousands of customer complaints from the Consumer Financial Protection Bureau (CFPB). The application features a multi-page Streamlit dashboard for interactive data analysis and conversational Q&A with an AI assistant grounded in real-world customer feedback.

## 🎯 Project Objective

To transform raw, unstructured complaint narratives into a strategic asset. This AI assistant empowers product, compliance, and support teams to:

*   **Identify major complaint trends** in minutes instead of days.
*   **Answer product-specific questions** with evidence-backed insights, without needing data analysts.
*   **Proactively discover and address** customer pain points before they escalate.

## 🏛️ System Architecture

The initial RAG pipeline prototype has been systematically upgraded to a showcase-ready application. Key enhancements were made across four core areas: the data and indexing foundation, the core RAG logic, the evaluation methodology, and the multi-page Streamlit application.

### Data Processing Pipeline

*   **Memory-Efficient Processing:** Refactored to process the large CSV file (51,497 complaints) in batches (`pd.read_csv(chunksize=...)`).
*   **Enriched Metadata:** Enhanced metadata for each chunk includes `Date received` and `Issue`, enabling advanced filtering.
*   **Chunking Strategy:** Uses a data-driven approach (`chunk_size=300, overlap=20`) to create focused, semantically meaningful text units.

### Advanced RAG Core

The "brain" of the system is a sophisticated `RAGPipeline` class that orchestrates several models:

*   **Upgraded Embedding Model:** Replaced `all-MiniLM-L6-v2` with `BAAI/bge-base-en-v1.5` for richer, more nuanced vector embeddings.
*   **Vector Store:** A FAISS index stores approximately 1.9 million vectors for high-speed retrieval.
*   **Cross-Encoder Re-ranker:** Added a re-ranking step using `cross-encoder/ms-marco-MiniLM-L-6-v2` to ensure the highest quality context is provided to the LLM.
*   **Sentiment Analysis:** Analyzes the sentiment of retrieved documents to provide a concise summary (e.g., "4 NEGATIVE, 1 POSITIVE"), fulfilling a key business objective.
*   **Advanced Prompt Engineering:** Uses a detailed template (`src/prompts.py`) to give the LLM a clear persona and strict instructions, leading to more reliable answers.
*   **Generative LLM:** `google/flan-t5-base` synthesizes the final, context-aware answer.

### Comprehensive Evaluation Framework

The evaluation method was overhauled to be more rigorous and data-driven.

*   **Formalized Evaluation Dataset:** Expanded into `evaluation/evaluation_dataset.csv` with over 30 diverse questions.
*   **Integrated Evaluation Workflow:**
    *   **Manual Qualitative Scoring:** An interactive loop for a human expert to provide a quality score (1-5) and comments.
    *   **Automated Quantitative Metrics:** Integration with the **RAGAs** framework to calculate scores for `faithfulness`, `answer_relevancy`, and `context_precision`.
*   **Unified Reporting:** Produces a single CSV file containing manual scores, comments, and all automated RAGAs scores.

### Multi-Page Application Architecture

A professional, multi-page Streamlit application provides a clean and intuitive user experience.

*   **Homepage:** A landing page with a high-level overview and key dataset statistics.
*   **Analysis Dashboard:** An interactive dashboard for exploratory data analysis with filters for date, products, and companies, visualizations, and PDF report generation.
*   **Complaint Chatbot Page:** Implements the RAG-based chatbot for evidence-backed Q&A, including an in-app feedback mechanism.

### Code Quality & Engineering Best Practices

*   **Modularization:** Code broken down into clean, reusable Python modules in the `src/` directory.
*   **Configuration Management:** All paths and parameters moved into a central `config.yaml` file.
*   **Robust Execution:** Implemented correct procedures for running long processes on Windows.

## 🛠️ Tech Stack

*   **Backend:** Python 3.10+, Pandas
*   **Core RAG Libraries:** sentence-transformers, faiss-cpu, transformers, langchain
*   **Evaluation:** ragas, datasets
*   **Frontend:** streamlit

### Key Models

*   **Embedding:** `BAAI/bge-base-en-v1.5`
*   **Re-ranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
*   **Generation:** `google/flan-t5-base`
*   **Sentiment:** `distilbert-base-uncased-finetuned-sst-2-english`

## 📁 Project Structure
rag_chatbot/
│
├── data/
│ ├── raw/
│ ├── clean/
│ │ └── complaints_clean.csv
│ ├── filtered_complaints.csv
│ ├── balanced_sampled_complaints.csv
│ └── vector_store/
│ ├── index_bge_base_300_20.faiss
│ └── meta_bge_base_300_20.csv
│
├── evaluation/
│ ├── evaluation_dataset.csv
│ ├── evaluation_results.csv
│ ├── generated_questions/
│ └── comprehensive_evaluation_results.csv
│
├── notebooks/
│ ├── task-1/
│ ├── task-2/
│ │ ├── chunking_evaluation.ipynb
│ │ └── embedding_and_indexing.ipynb
│ ├── task-3/
│ │ ├── rag_evaluation_results.csv
│ │ ├── rag_pipeline_evaluation.ipynb
│ │ └── rag_pipeline.ipynb
│ └── task-4/
│
├── pages/
│ ├── 2_💬_Chatbot.py
│ └── 3_📈_Analysis_and_Report.py
│
├── src/
│ ├── pycache/
│ ├── init.py
│ ├── chatbot_model.py
│ ├── chunking.py
│ ├── chunking_and_evaluate.py
│ ├── data_loader.py
│ ├── eda.py
│ ├── embedding.py
│ ├── embedding_indexing.py
│ ├── filter_clean.py
│ ├── indexing.py
│ ├── langchain_chunking.py
│ ├── prompts.py
│ ├── rag_logic.py
│ ├── RAG_pipeline_eval.py
│ └── rag_pipeline.py
│
├── 1_🏠_Homepage.py # Main Streamlit entry point
├── config.yaml # Central configuration file
├── run_indexing.py # Script to build the FAISS index
├── requirements.txt
└── README.md

text

## 🚀 How to Run the Application

### 1. Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd rag_chatbot

# Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
2. Prepare Data and Build Index
bash
# (Optional) Run preprocessing if you have raw data
# python src/preprocess_data.py

# (Optional) Create the balanced sample dataset
# python src/create_balanced_sample.py

# Run the indexing script (this will take 1-2 hours on a CPU)
python -u run_indexing.py
3. Launch the Streamlit App
bash
streamlit run app\app.py
📊 Evaluation Results Example
The comprehensive evaluation combines manual scores with automated RAGAs metrics.

Question	Generated Answer	Manual Score (1-5)	RAGAs Faithfulness	RAGAs Answer Relevancy
What issues are reported with credit card disputes?	Users report issues primarily related to billing disputes and unauthorized or fraudulent charges on their accounts.	5	1.00	0.98
Why do users complain about savings accounts?	Complaints often mention issues with fraud detection algorithms that lock accounts, and difficulties in accessing funds during investigations.	4	1.00	0.95
🔮 Future Work & Feature Plan
To transition from a reactive query system to a proactive insights platform.

Complaint Severity Classification: Implement a model to automatically tag complaints by urgency (e.g., Critical, High, Medium) using a pre-trained Hugging Face transformer pipeline for prioritization.

Proactive Trend & Anomaly Detection: Add time-series analysis to the dashboard to automatically detect and highlight unusual spikes in complaint volume using pandas and statistical deviation analysis.

Automated Topic Modeling: Integrate a library like BERTopic to generate weekly/monthly reports that automatically discover and summarize hidden, recurring complaint themes.