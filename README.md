
# 🧠 Intelligent Complaint Analysis for Financial Services

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot that helps internal teams at **CrediTrust Financial** understand customer complaints across five key product lines: Credit Cards, Personal Loans, Buy Now Pay Later (BNPL), Savings Accounts, and Money Transfers. It leverages complaint narratives from the Consumer Financial Protection Bureau (CFPB) to enable natural language querying of real-world customer feedback.

---

## 🚀 Project Objective

To transform thousands of raw, unstructured complaint narratives into a searchable and interactive AI assistant that empowers product, compliance, and support teams to:

- Identify major complaint trends in minutes instead of days.
- Answer product-specific questions without needing SQL or data analysts.
- Proactively address customer issues with contextual and evidence-backed insights.

---

## 🧱 Project Structure

```
INTELLIGENT-COMPLAINT-ANALYSIS/
│
├── data/
│   ├── raw/                       # Raw CFPB data
│   └── clean/
│       ├── complaints_clean.csv  # Cleaned narratives
│       └── vector_store/
│           ├── index_300_50.faiss     # FAISS index of embeddings
│           └── meta_300_50.csv        # Chunk metadata
│
├── notebooks/
│   ├── task-1/
│   │   ├── EDA_analysis.ipynb
│   │   └── loading_filtering.ipynb
│   ├── task-2/
│   │   ├── chunking_eval.ipynb
│   │   └── embedding_and_indexing.ipynb
│   ├── task-3/
│   │   └── rag_pipeline.ipynb
│   └── task-4/
│       └── (UI development notebook)
│
├── src/
│   ├── eda.py
│   ├── filter_clean.py
│   ├── chunking.py
│   ├── langchain_chunking.py
│   ├── embedding.py
│   ├── embedding_indexing.py
│   ├── vector_store.py
│   ├── indexing.py
│   └── rag_pipeline.py
│
├── python/
│   └── embaded_full.py           # Dev test script
│
├── requirements.txt              # Required Python packages
├── LICENSE
└── README.md
```

---

## ⚙️ Key Components

### 1. 📊 Data Cleaning & EDA
- Filtered for 5 key products.
- Removed null and boilerplate entries.
- Visualized word counts and complaint volume per product.

### 2. ✂️ Chunking Strategy
- Implemented using LangChain’s RecursiveCharacterTextSplitter.
- Final config: `chunk_size=300`, `chunk_overlap=50` → balance between context and chunk granularity.

### 3. 🔍 Embedding & Vector Indexing
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Vector store: FAISS with 384-d embeddings + metadata tracking.

### 4. 🧠 RAG Pipeline
- Retrieves top-k relevant complaint chunks for a query.
- Feeds them into an LLM (`google/flan-t5-base`) to generate grounded answers.
- Prompt tuned for contextual, non-hallucinated insights.

---

## 🧪 How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Preprocessing & Indexing**
   ```bash
   python src/filter_clean.py
   python src/embedding_indexing.py
   ```

3. **Test RAG Answering**
   ```bash
   python src/rag_pipeline.py
   ```

4. *(Optional)* Launch interactive UI (Gradio/Streamlit coming in Task-4).

---

## 💡 Sample Query

> ❓ Why are customers unhappy with BNPL?  
> 💬 “Many users complain about unexpected fees, poor transparency, and inability to manage installment schedules. The most common theme is dissatisfaction with the repayment process and unclear terms.”

---

## 📌 Tech Stack

- Python 3.10+
- SentenceTransformers (MiniLM-L6-v2)
- FAISS
- Hugging Face Transformers
- LangChain
- Pandas, NumPy, Matplotlib

---

## 📈 Project Status

- ✅ Task 1: Data cleaning, EDA complete
- ✅ Task 2: Chunking, embedding, FAISS indexing done
- ✅ Task 3: RAG pipeline and qualitative evaluation
- ⏳ Task 4: Interactive UI (in progress)

---

## 📄 License

This project is open-source and available under the [Apache License](LICENSE).
