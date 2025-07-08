#streamlit run streamlit_app.py
# task-4/streamlit_app.py

import streamlit as st
import pandas as pd
import os
import sys

# --- Add src directory to path ---
project_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from rag_pipeline import RAGPipeline

# --- Streamlit Setup ---
st.set_page_config(page_title="RAG QA Evaluator", layout="centered")
st.title("💡 Intelligent Complaint Analysis Assistant")
st.markdown("""
Ask a question based on customer complaints.
This assistant retrieves real complaint excerpts and uses them to generate an answer.
""")

# --- Load RAG pipeline ---
rag = RAGPipeline(top_k=5)

# --- Session state setup ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Input Box ---
st.subheader("🔍 Ask a New Question")
new_question = st.text_input("Type your question here:")
ask_button = st.button("🚀 Ask Another")
clear = st.button("🔄 Clear Conversation")

if clear:
    st.session_state.history = []
    st.rerun()


if ask_button and new_question:
    with st.spinner("Retrieving and answering..."):
        answer, chunks = rag.generate_answer(new_question)
        st.session_state.history.append({
            "question": new_question,
            "answer": answer,
            "chunks": chunks
        })
    st.rerun()


# --- Display Full Conversation History ---
if st.session_state.history:
    for idx, entry in enumerate(reversed(st.session_state.history), 1):
        q_num = len(st.session_state.history) - idx + 1
        st.markdown("---")
        st.subheader(f"❓ Question {q_num}:")
        st.markdown(f"**{entry['question']}**")

        st.subheader("💬 Answer")
        st.markdown(f"<div style='background-color:#f0f2f6; padding: 1rem; border-radius: 5px'>{entry['answer']}</div>", unsafe_allow_html=True)

        st.subheader("📄 Supporting Sources")
        for i, chunk in enumerate(entry['chunks'][:2]):
            with st.expander(f"Chunk {i+1}"):
                st.write(chunk)

        # Evaluation block per Q&A
        st.subheader("📝 Evaluate This Answer")
        score = st.radio(f"Rate Q{q_num} (1 = Poor, 5 = Excellent)", [1, 2, 3, 4, 5], key=f"score_{idx}", horizontal=True)
        comment = st.text_area("Comment (why this score?)", key=f"comment_{idx}")

        if st.button(f"✅ Save Evaluation for Q{q_num}", key=f"save_{idx}"):
            record = {
                "question": entry['question'],
                "answer": entry['answer'],
                "source_1": entry['chunks'][0] if len(entry['chunks']) > 0 else "",
                "source_2": entry['chunks'][1] if len(entry['chunks']) > 1 else "",
                "score": score,
                "comment": comment
            }

            if os.path.exists("evaluation_log.csv"):
                df_existing = pd.read_csv("evaluation_log.csv")
                df = pd.concat([df_existing, pd.DataFrame([record])], ignore_index=True)
            else:
                df = pd.DataFrame([record])

            df.to_csv("evaluation_log.csv", index=False)
            st.success(f"✅ Evaluation for Q{q_num} saved.")
