import streamlit as st
import os
import tempfile
import logging
from typing import List, Dict, Any
import pandas as pd
from pathlib import Path
import json

from main_rag_withoutShap import (
    get_hybrid_retriever_from_csv,
    run_rag_pipeline,
    clean_llm_output,
    EvidenceAttributor,
    TokenAttributor,
    CitationMetrics
)
from ingestion import update_knowledge_base, deduplicate_sources
from langchain_ollama import OllamaLLM

# Configuration
CSV_PATH = "educational_knowledge_base.csv"
ASSETS_DIR = Path("project_data/assets")

# Page configuration
st.set_page_config(
    page_title="EduRAG  ",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Comprehensive Dark Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Root dark background */
    .stApp {
        background-color: #0b0f19 !important;
        background-image: radial-gradient(circle at 50% 0%, #1e293b 0%, #0b0f19 70%);
    }
    
    .main {
        background-color: transparent !important;
    }
    
    .block-container {
        background-color: transparent !important;
        padding: 3rem 2rem !important;
    }
    
    /* Sidebar - Deep slate */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
        border-right: 1px solid #334155;
    }
    
    section[data-testid="stSidebar"] > div {
        background-color: transparent !important;
    }
    
    /* All text */
    p, span, label, h1, h2, h3, h4, h5, h6, div {
        color: #e2e8f0 !important;
    }
    
    h1 { color: #38bdf8 !important; font-weight: 700; }
    h2, h3 { color: #60a5fa !important; font-weight: 600; }
    
    /* Tabs - Dark theme */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1e293b !important;
        border-radius: 8px !important;
        padding: 8px !important;
        border: 1px solid #334155 !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8 !important;
        background-color: transparent !important;
        border: none !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: #38bdf8 !important;
        background-color: #0f172a !important;
        border-radius: 6px !important;
        border: 1px solid #334155 !important;
    }
    
    /* Inputs */
    div[data-baseweb="input"] > div,
    .stTextInput > div > div,
    .stTextArea > div > div {
        background-color: #1e293b !important;
        border: 1px solid #475569 !important;
        color: #f1f5f9 !important;
        border-radius: 8px !important;
    }
    
    input, textarea {
        color: #f1f5f9 !important;
    }
    
    input::placeholder {
        color: #94a3b8 !important;
    }
    
    /* Selectbox */
    div[data-baseweb="select"] > div,
    .stSelectbox > div > div {
        background-color: #1e293b !important;
        border: 1px solid #475569 !important;
        color: #f1f5f9 !important;
    }
    
    /* File uploader */
    .stFileUploader > div > div {
        background-color: #1e293b !important;
        border: 2px dashed #475569 !important;
        color: #e2e8f0 !important;
    }
    
    .stFileUploader > div > div:hover {
        border-color: #3b82f6 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.4) !important;
    }
    
    /* Secondary button */
    button[kind="secondary"] {
        background: #334155 !important;
        color: #e2e8f0 !important;
        border: 1px solid #475569 !important;
    }
    
    /* Toggle */
    .stToggle > div > div {
        background-color: #334155 !important;
    }
    
    .stToggle > div > div > div {
        background-color: #38bdf8 !important;
    }
    
    /* Checkbox */
    .stCheckbox > div > div > div {
        background-color: #1e293b !important;
        border-color: #475569 !important;
    }
    
    /* Cards */
    div[data-testid="stHorizontalBlock"] > div > div {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        padding: 24px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        padding: 20px !important;
    }
    
    div[data-testid="stMetric"] > label {
        color: #94a3b8 !important;
        font-size: 0.85em !important;
        text-transform: uppercase !important;
    }
    
    div[data-testid="stMetric"] .css-1xarl3l {
        color: #38bdf8 !important;
        font-size: 1.8em !important;
        font-weight: 700 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
    }
    
    .streamlit-expanderContent {
        background-color: #0f172a !important;
        border: 1px solid #334155 !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
    }
    
    /* Answer box */
    .answer-box {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%) !important;
        border-left: 4px solid #60a5fa !important;
        border-radius: 12px !important;
        padding: 28px !important;
        margin: 16px 0 !important;
        box-shadow: 0 10px 15px -3px rgba(30, 64, 175, 0.3) !important;
    }
    
    .answer-box p {
        color: #f8fafc !important;
        font-size: 1.15em !important;
        line-height: 1.7 !important;
        margin: 0 !important;
    }
    
    /* Source cards */
    .source-card {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        padding: 20px !important;
        margin-bottom: 16px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Attribution bars */
    .attribution-bar {
        height: 10px !important;
        background-color: #334155 !important;
        border-radius: 5px !important;
        overflow: hidden !important;
        margin-top: 8px !important;
    }
    
    .attribution-fill {
        height: 100% !important;
        background: linear-gradient(90deg, #3b82f6, #06b6d4) !important;
        border-radius: 5px !important;
        box-shadow: 0 0 10px rgba(59, 130, 246, 0.5) !important;
    }
    
    /* Token heatmap */
    .token-grounded {
        background-color: #065f46 !important;
        color: #6ee7b7 !important;
        padding: 4px 8px !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        border: 1px solid #10b981 !important;
    }
    
    .token-ungrounded {
        background-color: #7f1d1d !important;
        color: #fca5a5 !important;
        padding: 4px 8px !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        border: 1px solid #ef4444 !important;
    }
    
    .token-partial {
        background-color: #92400e !important;
        color: #fcd34d !important;
        padding: 4px 8px !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        border: 1px solid #f59e0b !important;
    }
    
    /* Badges */
    .badge-success {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
        color: #ffffff !important;
        padding: 8px 16px !important;
        border-radius: 20px !important;
        font-size: 0.9em !important;
        font-weight: 700 !important;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.4) !important;
        border: 1px solid #34d399 !important;
    }
    
    .badge-warning {
        background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%) !important;
        color: #ffffff !important;
        padding: 8px 16px !important;
        border-radius: 20px !important;
        font-size: 0.9em !important;
        font-weight: 700 !important;
        box-shadow: 0 0 15px rgba(245, 158, 11, 0.4) !important;
    }
    
    .badge-danger {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%) !important;
        color: #ffffff !important;
        padding: 8px 16px !important;
        border-radius: 20px !important;
        font-size: 0.9em !important;
        font-weight: 700 !important;
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.4) !important;
    }
    
    /* Dataframe */
    .stDataFrame {
        background-color: #1e293b !important;
    }
    
    .stDataFrame td, .stDataFrame th {
        color: #e2e8f0 !important;
        border-color: #334155 !important;
        background-color: #1e293b !important;
    }
    
    /* Image caption */
    figcaption {
        color: #94a3b8 !important;
        background-color: #1e293b !important;
        padding: 8px !important;
        border-radius: 0 0 8px 8px !important;
    }
    
    /* Dividers */
    hr {
        border-color: #334155 !important;
        margin: 2rem 0 !important;
    }
    
    /* Scrollbars */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0f172a !important;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #475569 !important;
        border-radius: 5px !important;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #64748b !important;
    }
    
    /* Code blocks */
    .stCode pre {
        background-color: #0f172a !important;
        border: 1px solid #334155 !important;
        color: #e2e8f0 !important;
    }
    
    /* Alerts */
    .stAlert {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
    }
    
    /* Info box */
    .stInfo {
        background-color: #1e3a8a !important;
        border: 1px solid #3b82f6 !important;
        color: #e0f2fe !important;
    }
    
    /* Warning box */
    .stWarning {
        background-color: #7c2d12 !important;
        border: 1px solid #f59e0b !important;
        color: #fef3c7 !important;
    }
    
    /* Error box */
    .stException {
        background-color: #7f1d1d !important;
        border: 1px solid #ef4444 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'show_attribution' not in st.session_state:
    st.session_state.show_attribution = True
if 'show_tokens' not in st.session_state:
    st.session_state.show_tokens = True

# Initialize RAG
@st.cache_resource
def load_pipeline(enable_hyde=True):
    try:
        with st.spinner("🧠 Initializing neural networks..."):
            retriever = get_hybrid_retriever_from_csv(CSV_PATH, top_k=5, enable_hyde=enable_hyde)
            return retriever
    except Exception as e:
        st.error(f"❌ Initialization failed: {e}")
        return None

# Sidebar with Stats and Settings
with st.sidebar:
    st.markdown("# 🎓 EduRAG ")
    
    
    st.markdown("---")
    
    # Knowledge Base Stats
    st.header("📊 Knowledge Base Stats")
    
    if os.path.exists(CSV_PATH):
        try:
            df_stats = pd.read_csv(CSV_PATH)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Sources", len(df_stats['source_file'].unique()))
            with col2:
                st.metric("Chunks", len(df_stats))
            
            col3, col4 = st.columns(2)
            with col3:
                text_count = len(df_stats[df_stats['type']=='text'])
                st.metric("Text", text_count)
            with col4:
                visual_count = len(df_stats[df_stats['type']=='visual_content'])
                st.metric("Visuals", visual_count)
            
            # Source breakdown
            with st.expander("View Sources"):
                sources = df_stats['source_file'].unique()
                for src in sources[:5]:
                    st.markdown(f"• `{src}`")
                if len(sources) > 5:
                    st.caption(f"... and {len(sources)-5} more")
        except Exception as e:
            st.error(f"Error loading stats: {e}")
    else:
        st.warning("⚠️ No knowledge base found")
    
    st.markdown("---")
    
    # Global Settings
    st.header("⚙️ Settings")
    
    enable_hyde = st.toggle("Enable HyDE", value=True, 
                           help="Hypothetical Document Embeddings improve retrieval accuracy")
    
    st.markdown("---")
    
    # Model Status
    st.header("🤖 Model Status")
    st.markdown("• **LLM**: DeepSeek R1 8B")
    st.markdown("• **Embeddings**: all-MiniLM-L6-v2")
    st.markdown("• **VLM**: LLaVA-Phi3")
    
    st.markdown("---")
    st.caption("🔒 Local Processing • Privacy First")

# Load pipeline
if st.session_state.pipeline is None:
    st.session_state.pipeline = load_pipeline(enable_hyde=enable_hyde)

retriever = st.session_state.pipeline

# Main Tabs
qa_tab, upload_tab, viz_tab = st.tabs(["📚 QA Assistant", "⬆️ Upload Sources", "🔍 Knowledge Map"])

# ====================================================
# TAB 1: QA Assistant
# ====================================================
with qa_tab:
    st.title("📚 Multimodel Educational Assistant")
    st.markdown("Ask questions with citation grounding and faithfulness validation")
    
    if retriever:
        # Query layout
        col1, col2 = st.columns([3, 1])
        with col1:
            user_query = st.text_input(
                "Enter your question:",
                placeholder="e.g., 'Explain the diagram on page 42 about database architecture'"
            )
        with col2:
            search_type = st.selectbox(
                "Search mode:",
                ["Auto", "Text Only", "Visual/Diagrams"],
                help="Force retrieval of visual content for diagram questions"
            )
        
        # Action buttons
        btn_col1, btn_col2 = st.columns([1, 5])
        with btn_col1:
            submit = st.button("🔍 Submit", use_container_width=True, type="primary")
        
        if submit and user_query.strip():
            with st.spinner("Retrieving and analyzing..."):
                try:
                    # Modify query if visual mode selected
                    query_input = user_query
                    if search_type == "Visual/Diagrams":
                        query_input = f"[VISUAL] {user_query}"
                    
                    # Run pipeline
                    result = run_rag_pipeline(query_input, retriever, llm=None)
                    
                    # Display Answer
                    st.markdown("### 💡 Generated Answer")
                    
                    ans_col, meta_col = st.columns([2, 1])
                    
                    with ans_col:
                        st.markdown(f"""
                        <div class="answer-box">
                            <p>{result.answer}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.caption(f"Retrieved {len(result.context)} sources")
                    
                    with meta_col:
                        # Reliability badge
                        reliability = (result.faithfulness + result.retrieval_score + result.consensus.get('consensus_score', 0)) / 3
                        
                        if reliability > 0.8:
                            st.markdown('<div class="badge-success">✅ High Reliability</div>', unsafe_allow_html=True)
                        elif reliability > 0.6:
                            st.markdown('<div class="badge-warning">⚠️ Moderate</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="badge-danger">❌ Verify</div>', unsafe_allow_html=True)
                        
                        st.metric("Faithfulness", f"{result.faithfulness:.2f}")
                        st.metric("Citation Valid", f"{result.citation_metrics.get('valid_citation_rate', 0)*100:.0f}%")
                    
                    # Detailed Metrics
                    with st.expander("📊 Detailed Reliability Metrics", expanded=True):
                        met_col1, met_col2, met_col3, met_col4 = st.columns(4)
                        with met_col1:
                            st.metric("Retrieval", f"{result.retrieval_score:.3f}")
                        with met_col2:
                            st.metric("Faithful", f"{result.faithfulness:.3f}")
                        with met_col3:
                            con = result.consensus.get('consensus_score', 0)
                            st.metric("Consensus", f"{con:.3f}")
                        with met_col4:
                            gr = result.token_attribution.get('grounding_ratio', 0)
                            st.metric("Grounding", f"{gr*100:.1f}%")
                        
                        # Consensus warning
                        if result.consensus.get('status') == "CONFLICT_DETECTED":
                            st.error("⚠️ Sources may contradict. Verify claims.")
                            for conflict in result.consensus.get('conflicts', [])[:2]:
                                st.caption(f"Conflict: {conflict['doc_a']['id']} vs {conflict['doc_b']['id']}")
                    
                    # Evidence Attribution
                    if st.session_state.show_attribution and result.evidence_attribution.get('attributions'):
                        st.markdown("### 🔍 Evidence Attribution")
                        
                        attrs = result.evidence_attribution['attributions'][:3]
                        cols = st.columns(len(attrs))
                        
                        for idx, attr in enumerate(attrs):
                            with cols[idx]:
                                pct = attr['contribution_pct']
                                st.markdown(f"""
                                <div style="background: #1e293b; padding: 16px; border-radius: 8px; border: 1px solid #334155;">
                                    <div style="font-weight: 600; color: #38bdf8; margin-bottom: 4px;">
                                        📄 {attr['source'][:20]}...
                                    </div>
                                    <div style="font-size: 0.8em; color: #94a3b8; margin-bottom: 8px;">
                                        Page {attr['page']}
                                    </div>
                                    <div class="attribution-bar">
                                        <div class="attribution-fill" style="width: {pct}%;"></div>
                                    </div>
                                    <div style="text-align: right; color: #38bdf8; font-weight: 600; margin-top: 4px;">
                                        {pct:.1f}%
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Token Grounding
                    if st.session_state.show_tokens and result.token_attribution.get('tokens'):
                        st.markdown("### 🔤 Token Grounding Heatmap")
                        st.caption("Green = grounded • Yellow = partial • Red = ungrounded")
                        
                        html = ""
                        for t in result.token_attribution['tokens'][:50]:  # Limit display
                            token = t['token']
                            score = t['grounded_score']
                            
                            if score > 0.8:
                                cls = "token-grounded"
                            elif score > 0.5:
                                cls = "token-partial"
                            else:
                                cls = "token-ungrounded"
                            
                            html += f'<span class="{cls}">{token}</span> '
                        
                        st.markdown(f"""
                        <div style="background: #1e293b; padding: 20px; border-radius: 8px; border: 1px solid #334155; line-height: 2.2;">
                            {html}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Sources
                    st.markdown("### 📚 Evidence Sources")
                    
                    text_docs = [d for d in result.context if d.metadata.get("type") != "visual_content"]
                    visual_docs = [d for d in result.context if d.metadata.get("type") == "visual_content"]
                    
                    if visual_docs:
                        st.markdown("#### 🖼️ Visual References")
                        img_cols = st.columns(min(len(visual_docs), 3))
                        for idx, doc in enumerate(visual_docs):
                            with img_cols[idx % 3]:
                                img_path = doc.metadata.get("image_ref")
                                if img_path and os.path.exists(img_path):
                                    st.image(
                                        img_path, 
                                        caption=f"Page {doc.metadata.get('page_number')} • {doc.metadata.get('source_file', '')[:15]}"
                                    )
                                else:
                                    st.error("Image not found")
                    
                    if text_docs:
                        st.markdown("#### 📝 Text Sources")
                        for i, doc in enumerate(text_docs, 1):
                            st.markdown(f"""
                            <div class="source-card">
                                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px; color: #60a5fa; font-weight: 600; font-size: 0.9em;">
                                    <span style="background: #3b82f6; color: white; padding: 2px 8px; border-radius: 4px;">[{i}]</span>
                                    <span>{doc.metadata.get('source_file', 'Unknown')}</span>
                                    <span style="color: #475569;">|</span>
                                    <span>Page {doc.metadata.get('page_number', 'N/A')}</span>
                                    <span style="color: #475569;">|</span>
                                    <span>{doc.metadata.get('section_path', 'General')}</span>
                                </div>
                                <div style="color: #cbd5e1; font-size: 0.95em; line-height: 1.5;">
                                    {doc.page_content[:300]}{"..." if len(doc.page_content) > 300 else ""}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error processing query: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    else:
        st.error("RAG system not loaded. Check if Ollama is running and KB exists.")

# ====================================================
# TAB 2: Upload Sources
# ====================================================
with upload_tab:
    st.title("⬆️ Knowledge Base Management")
    
    # Advanced options
    if st.checkbox("Show Advanced Options"):
        vlm_enabled = st.toggle("Enable VLM Image Description (LLaVA)", value=True)
        st.info("VLM generates searchable text from diagrams. Requires Ollama with llava-phi3.")
    else:
        vlm_enabled = True
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload Documents (PDF, DOCX, PPTX):", 
        type=["pdf", "docx", "pptx"],
        accept_multiple_files=True
    )
    
    # URL input
    url_input = st.text_area(
        "Or enter URLs (one per line):", 
        placeholder="https://example.com/textbook.pdf"
    )
    
    # Process button
    if st.button("Process & Update Knowledge Base", type="primary"):
        sources_to_process = []
        
        # Handle uploaded files
        if uploaded_files:
            with tempfile.TemporaryDirectory() as tmpdir:
                for uploaded_file in uploaded_files:
                    temp_path = os.path.join(tmpdir, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    sources_to_process.append(temp_path)
                
                # Handle URLs
                if url_input.strip():
                    sources_to_process.extend([url.strip() for url in url_input.split('\n') if url.strip()])
                
                if not sources_to_process:
                    st.warning("No sources provided.")
                else:
                    # Process
                    with st.spinner(f"Processing {len(sources_to_process)} sources... This may take several minutes."):
                        try:
                            total_chunks = update_knowledge_base(
                                sources_to_process,
                                kb_path=CSV_PATH,
                                assets_dir=str(ASSETS_DIR),
                                vlm_enabled=vlm_enabled
                            )
                            
                            # Clear cache to force reload
                            load_pipeline.clear()
                            
                            st.success(f"✅ Successfully added content! Total KB size: {total_chunks} chunks")
                            st.balloons()
                            st.info("🔄 Please refresh the page to reload the updated knowledge base.")
                            
                        except Exception as e:
                            st.error(f"Processing failed: {e}")
                            st.exception(e)

# ====================================================
# TAB 3: Knowledge Visualization
# ====================================================
with viz_tab:
    st.title("🔍 Knowledge Base Explorer")
    
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        
        # Filter by source
        sources = df['source_file'].unique().tolist()
        selected_source = st.selectbox("Select Document:", ["All"] + sources)
        
        if selected_source != "All":
            df = df[df['source_file'] == selected_source]
        
        # Show sections
        sections = df['section_path'].unique().tolist()
        st.markdown(f"**Sections in this document:** {len(sections)}")
        
        selected_section = st.selectbox("Navigate to Section:", sections[:20])
        
        if selected_section:
            section_data = df[df['section_path'] == selected_section]
            st.markdown(f"**Chunks in section:** {len(section_data)}")
            
            # Display chunks
            for _, row in section_data.iterrows():
                with st.container():
                    if row['type'] == 'visual_content' and row['image_ref'] and os.path.exists(row['image_ref']):
                        st.image(row['image_ref'], caption=f"Page {row['page_number']}", width=300)
                    else:
                        # Text preview
                        chunk_id_display = row.get('chunk_id', 'Unknown') if hasattr(row, 'get') else row['chunk_id'] if 'chunk_id' in row else 'Unknown'
                        page_number_display = row.get('page_number', 'N/A') if hasattr(row, 'get') else row['page_number'] if 'page_number' in row else 'N/A'
                        st.markdown(f"""
                        <div style="background: #1e293b; padding: 16px; border-radius: 8px; border: 1px solid #334155; margin-bottom: 12px;">
                            <div style="color: #60a5fa; font-size: 0.85em; font-weight: 600; margin-bottom: 8px;">
                                {chunk_id_display} • Page {page_number_display}
                            </div>
                            <div style="color: #e2e8f0; font-size: 0.95em;">
                                {str(row.get('text', ''))[:200]}{"..." if len(str(row.get('text', ''))) > 200 else ""}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
    else:
        st.info("Upload documents to see knowledge map")

# Footer
st.markdown("---")
st.caption("EduRAG | Docling + Ollama + LangChain")
