import streamlit as st
import os
import tempfile
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# --- REQUIRED IMPORTS FROM PROJECT FILES ---
# NOTE: Ensure textbook_preprocessor_with_indexes.py defines the process_documents function
from main_rag_app import (
    get_hybrid_retriever_from_csv,
    get_qa_chain,
    clean_llm_output,
    compute_faithfulness,
    compute_plausibility,
    interpret_rag_scores,
)
from textbook_preprocessor_with_indexes import process_documents # <-- REQUIRED IMPORT
from langchain_ollama import OllamaLLM

# --- Configuration ---
CSV_PATH = "educational_knowledge_base.csv"

# --- RAG System Initialization (Caches retriever and chain) ---
@st.cache_resource
def load_pipeline():
    """Initializes the RAG system components (Retriever, QA Chain, Scoring LLM)."""
    try:
        logging.info("Initializing RAG system (Retriever and QA Chain)...")
        
        # 1. Initialize retriever
        retriever = get_hybrid_retriever_from_csv(CSV_PATH, top_k=5)
        
        # 2. Initialize QA chain
        qa_chain = get_qa_chain(retriever)
        
        # 3. Initialize LLM for scoring (Plausibility)
        llm_score = OllamaLLM(model="deepseek-r1:1.5b")
        
        logging.info("RAG system initialized successfully.")
        return retriever, qa_chain, llm_score
    except Exception as e:
        logging.error(f"Failed to initialize RAG system: {e}")
        st.error(f"RAG system initialization failed. Check logs, Ollama status, and '{CSV_PATH}' file existence. Error: {e}")
        return None, None, None # Return None on failure

retriever, qa_chain, llm_score = load_pipeline()

# --- Streamlit UI ---
st.set_page_config(page_title="RAG Educational Assistant", layout="wide")

# ----------------------------------------------------
# TAB IMPLEMENTATION
# ----------------------------------------------------
# We create two tabs and wrap the original QA content inside the first tab.
qa_tab, upload_tab = st.tabs(["📚 QA Assistant", "⬆️ Source Uploader"])

# ====================================================
# 1. QA Assistant Tab (Restored Original Interface)
# ====================================================
with qa_tab:
    st.title("📚 RAG Educational Assistant")
    st.markdown("Ask any question from your study materials. The system retrieves, analyzes, and explains the answer.")
    
    if qa_chain and retriever and llm_score:
        user_query = st.text_input("Enter your question:", key="qa_input")
        
        if st.button("Submit", key="qa_submit") and user_query.strip():
            with st.spinner("Thinking..."):
                # 1. Invoke the RAG chain
                raw = qa_chain.invoke({"input": user_query})
                
                answer_text = raw.get("answer", "No answer found.")
                context_docs = raw.get("context") or []

                # 2. Clean answer
                cleaned_answer = clean_llm_output(answer_text)

                # 3. Metrics Calculation
                retrieval_score = float(getattr(retriever, "last_retrieval_score", 0.0))
                faithfulness = compute_faithfulness(answer_text, context_docs) if context_docs else 0.0
                plausibility = compute_plausibility(llm_score, answer_text, user_query)

                interpretation = interpret_rag_scores(
                    faithfulness, plausibility, retrieval_score
                )

                shap_exp = getattr(retriever, "last_shap_explanation", "")

            # 4. UI Output
            st.subheader("📘 Final Answer")
            st.success(cleaned_answer)

            st.subheader("🧠 Reliability Check")
            st.write(interpretation)

            col1, col2, col3 = st.columns(3)
            col1.metric("Faithfulness", f"{faithfulness:.3f}")
            col2.metric("Plausibility", f"{plausibility:.3f}")
            col3.metric("Retrieval Score", f"{retrieval_score:.3f}")

            if shap_exp:
                st.subheader("🔍 SHAP Explanation")
                st.info(shap_exp)

            st.subheader("📄 Sources Used")
            for i, doc in enumerate(context_docs, 1):
                snippet = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                st.text_area(
                    f"Source {i} - {doc.metadata.get('source_file', 'N/A')} (Page {doc.metadata.get('page_number', 'N/A')}, ID: {doc.metadata.get('chunk_id', 'N/A')})",
                    value=snippet,
                    height=150
                )
    else:
        st.error("The RAG system is not loaded. Please check the terminal logs for initialization errors.")

# ====================================================
# 2. Source Uploader Tab (New Functionality)
# ====================================================
with upload_tab:
    st.title("⬆️ Knowledge Base Management")
    st.info(f"The knowledge base CSV is: **{CSV_PATH}**")
    st.warning("Uploading and processing new files can take several minutes. You **MUST** restart the Streamlit app after processing to load the new documents.")
    
    uploaded_files = st.file_uploader(
        "Upload PDF Files for the Knowledge Base:", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process & Update Knowledge Base", key="process_button"):
            
            with tempfile.TemporaryDirectory() as tmpdir:
                
                file_paths = []
                for uploaded_file in uploaded_files:
                    temp_path = os.path.join(tmpdir, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(temp_path)

                with st.spinner(f"Processing {len(file_paths)} files and appending to the knowledge base..."):
                    try:
                        total_chunks = process_documents(file_paths, output_csv=CSV_PATH)
                        
                        st.success(f"Successfully processed {len(file_paths)} file(s) and updated the knowledge base!")
                        st.markdown(f"**Total chunks now in KB:** `{total_chunks}`")
                        
                        # Invalidate the cache to force RAG re-initialization on next run
                        load_pipeline.clear() 
                        st.error("⚠️ **RAG system needs re-initialization.** Please **close the Streamlit app** in your terminal (Ctrl+C) and **re-run** `streamlit run app.py` to load the new sources.")
                        
                    except Exception as e:
                        st.error(f"An error occurred during processing: {e}")
                        st.exception(e)