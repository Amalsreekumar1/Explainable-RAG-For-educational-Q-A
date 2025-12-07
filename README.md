# RAG Educational Assistant 

A **Retrieval-Augmented Generation (RAG)** web app built with Streamlit for querying educational PDFs (e.g., textbooks). It processes documents into a searchable knowledge base, retrieves relevant chunks using hybrid search (BM25 + FAISS + RRF), generates answers with Ollama LLM, and provides reliability metrics (faithfulness, plausibility) plus SHAP explainability.

Perfect for students/teachers: Upload PDFs, ask questions, get cited answers with confidence scores. No cloud costs—runs locally!

## 🎯 Key Features
- **PDF Knowledge Base Builder**: Chunk and index textbooks into a CSV-based vector store (handles duplicates, headers/footers).
- **Hybrid Retrieval**: Combines keyword (BM25) + semantic (FAISS embeddings) search with reciprocal rank fusion (RRF) and embedding reranking.
- **LLM-Powered QA**: Uses Ollama (e.g., DeepSeek-R1:1.5B) for concise, context-grounded answers.
- **Reliability Dashboard**: Real-time scores for faithfulness (BERTScore), plausibility (LLM-judged), and retrieval quality (cosine sim).
- **Explainability**: SHAP values explain why certain chunks were prioritized (student-friendly + dev breakdowns).
- **Batch Evaluation**: Automated testing with NLP metrics (BERTScore, ROUGE, BLEU) on a QA dataset.
- **Streamlit UI**: Two tabs—QA chat + document uploader. Responsive and intuitive.

| Feature | Benefit |
|---------|---------|
| Local-First | Privacy-focused; no API keys needed beyond Ollama. |
| Extensible | Easy to swap LLMs/embeddings; add more metrics. |
| Efficient | Processes 100+ page PDFs in minutes. |

## 🏗️ Architecture Overview
User Query → HybridRetriever (BM25 + FAISS + RRF + Embedding Rerank + SHAP)

↓

Context Stuffing → Ollama LLM (Prompt: QA_TEMPLATE) → Cleaned Answer

↓

Metrics: Faithfulness (BERTScore vs. Context) + Plausibility (LLM Score) + Retrieval Sim

↓

UI: Answer + Interpretation + Metrics + Sources + SHAP Explainer


- **Core Files**:
  - `app.py`: Streamlit frontend (QA tab + uploader).
  - `main_rag_app.py`: RAG pipeline (retriever, chain, metrics, SHAP).
  - `textbook_preprocessor_with_indexes.py`: PDF extraction (PyMuPDF/pdfplumber), cleaning, chunking (sentence-aware, ~1000 chars).
  - `evaluation.py`: Batch eval script (uses `evaluate` lib for NLP scores).
- **Data Flow**: PDFs → Chunks (CSV: `knowledge_base.csv`) → Indexing → Retrieval → Generation.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+.
- [Ollama](https://ollama.com) installed and running (local LLM server).
- Git.

### Installation
1. Clone the repo:
   - git clone https://github.com/yourusername/rag-educational-assistant.git
   - cd rag-educational-assistant

2. Create a virtual environment (recommended):
   - python -m venv venv
   - source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:
    - pip install -r requirements.txt
      
    *Note*: This includes LangChain, FAISS, SHAP, Evaluate, and PDF libs. If you hit CUDA issues (e.g., for Torch), use CPU versions or install CUDA toolkit.

4. Set up Ollama:
   - Download/run Ollama: `ollama serve`.
   - Pull the model: `ollama pull deepseek-r1:1.5b` (or your preferred; edit in `main_rag_app.py`).
   - Ensure it's running on `http://localhost:11434`.

5. Configure (optional):
   - Copy `.env.example` to `.env` and edit (e.g., `HF_EMBEDDING_MODEL=all-MiniLM-L6-v2` for SentenceTransformers).

### Running the App
1. Process sample PDFs (if you have a `textbooks/` folder):
   python textbook_preprocessor_with_indexes.py --textbooks-dir textbooks --output educational_knowledge_base.csv
   *This builds `educational_knowledge_base.csv` with indexed chunks.*

2. Launch the Streamlit app: streamlit run app.py
   - Open `http://localhost:8501`.
   - **QA Tab**: Ask questions (e.g., "What is photosynthesis?").
   - **Uploader Tab**: Add new PDFs; restart app after processing.

### Evaluation
1. Prepare `evaluation_dataset.csv` (columns: `query`, `answer`—ground truth QA pairs).
2. Run: python evaluation.py
3. Outputs: Console summary + `evaluation_results_detailed.csv` with per-query scores.

## 📊 Example Usage
- **Query**: "Explain Newton's laws."
- **Output**:
- **Answer**: "Newton's first law states that an object at rest stays at rest... [3 sentences]."
- **Reliability**: "Highly reliable (avg score: 0.85)."
- **Metrics**: Faithfulness: 0.92 | Plausibility: 0.88 | Retrieval: 0.90.
- **SHAP**: "Chunk X influenced most due to direct keyword match."
- **Sources**: Snippets from `physics_textbook.pdf` (Page 45, Chunk ID: physics_001).

Demo GIF: [Insert your screen recording here via GitHub upload].

## 🔧 Troubleshooting
- **Ollama Errors**: Check `ollama list`; ensure model is pulled and server runs.
- **Import Failures**: Run `pip install -r requirements.txt --upgrade`; verify Torch/FAISS CPU compat.
- **PDF Processing Slow**: Use fewer pages or switch to PyMuPDF (faster than pdfplumber).
- **No Chunks in KB**: Ensure PDFs aren't scanned images (needs OCR—future enhancement).
- **SHAP Slow**: Disable with `EXPLAIN_WITH_SHAP=False` in `main_rag_app.py`.
- **Metrics Zero**: Install `evaluate` datasets: `pip install datasets`.

Common Issues Table:
| Issue | Cause | Fix |
|-------|-------|-----|
| "CSV not found" | No KB built | Run preprocessor first. |
| Low Faithfulness | Weak embeddings | Try `all-mpnet-base-v2`. |
| App Crashes on Upload | Large PDF | Increase Streamlit timeout or chunk size. |

## 🤝 Contributing
1. Fork the repo.
2. Create a feature branch: `git checkout -b feature/amazing-feature`.
3. Commit: `git commit -m "Add amazing feature"`.
4. Push: `git push origin feature/amazing-feature`.
5. Open a Pull Request.

Guidelines:
- Follow PEP8 (use Black: `pip install black; black .`).
- Add tests for new features.
- Update README with examples.

## 📄 License
MIT License—feel free to use/modify/share. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments
- [LangChain](https://langchain.com) for RAG primitives.
- [Ollama](https://ollama.com) for local inference.
- [Streamlit](https://streamlit.io) for the slick UI.
- Inspired by educational tools like Khan Academy + Perplexity AI.

Questions? Open an [issue](https://github.com/yourusername/rag-educational-assistant/issues) or DM me!

⭐ Star if this helps your studies! 🌟
