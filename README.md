# Explainable RAG For Educational Q A

Final year research project — currently in active development. Expected completion: May 2026

A cutting-edge **Retrieval-Augmented Generation (RAG)** system designed specifically for educational purposes. EduRAG processes textbooks, research papers, and lecture materials to create a searchable knowledge base that powers intelligent question-answering with advanced explainability features.

## 🎯 Key Features

### 🧠 Intelligent Retrieval
- **Hybrid Search Engine**: Combines BM25 keyword matching with dense semantic embeddings (BGE)
- **Reciprocal Rank Fusion**: Optimally blends different retrieval strategies
- **Cross-Encoder Reranking**: Ensures highest relevance for top results
- **Hypothetical Document Embeddings (HyDE)**: Improves retrieval for complex queries

### 🤖 Advanced AI Generation
- **Strict Grounding**: Answers based only on retrieved evidence
- **Citation Tracking**: Every claim is properly sourced [1], [2], etc.
- **Hallucination Detection**: Identifies ungrounded statements automatically

### 🔍 Comprehensive Explainability
- **Evidence Attribution**: Shows which sources contributed most to the answer
- **Token-Level Grounding**: Highlights grounded vs. ungrounded words in answers
- **Source Consensus Analysis**: Detects conflicting information in sources
- **Citation Validation**: Ensures all citations point to actual retrieved content

### 🖼️ Multimodal Support
- **Visual Content Processing**: Extracts and describes diagrams, charts, and images
- **Table Recognition**: Converts complex tables to searchable content
- **Document Structure Preservation**: Maintains chapters, sections, and hierarchies

### 📊 Evaluation & Benchmarking
- **Automated Ground Truth Generation**: Creates evaluation datasets from knowledge base
- **Comprehensive Metrics**: BERTScore, ROUGE, Faithfulness, and Retrieval scores
- **Citation Quality Analysis**: Measures precision and recall of citations

## 🏗️ Architecture Overview

```
User Query → HybridRetriever (BM25 + FAISS + RRF + Reranker + HyDE)
     ↓
Evidence Documents → Cross-Encoder Attribution & Token Grounding
     ↓
Strict Grounding Prompt → Ollama LLM (DeepSeek R1 8B)
     ↓
Answer + Citations + Explainability Metrics
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) installed and running
- Git

### Installation
```bash
# Clone the repository
git clone https://github.com/Amalsreekumar1/Explainable-RAG-For-educational-Q-A.git
cd Explainable-RAG-For-educational-Q-A

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Setup Models
```bash
# Pull required Ollama models
ollama pull deepseek-r1:8b
ollama pull deepseek-r1:1.5b
ollama pull llava-phi3
```

### Run the Application
```bash
# Start the Streamlit interface
streamlit run app.py
```

## 📚 Usage Guide

### 1. Building Your Knowledge Base
Place your educational materials (PDFs, DOCX, PPTX) in a folder and process them:

```bash
python ingestion.py
```

Or use the Upload Sources tab in the web interface.

### 2. Generating Evaluation Dataset
Create ground truth QA pairs for benchmarking:

```bash
python generate_ground_truth.py
```

### 3. Running Evaluations
Benchmark your system performance:

```bash
python evaluation.py
```

### 4. Command-Line Interface
Test the system directly in terminal:

```bash
python main_rag.py
```

## 🎛️ Web Interface Features

### 📚 QA Assistant Tab
- Natural language questioning with citation-grounded answers
- Real-time reliability scoring and metrics
- Evidence attribution visualization
- Token grounding heatmap
- Visual content display

### ⬆️ Upload Sources Tab
- Multi-format document processing (PDF, DOCX, PPTX)
- Automatic visual content extraction and description
- Incremental knowledge base updates
- Duplicate detection and prevention

### 🔍 Knowledge Map Tab
- Interactive exploration of processed documents
- Section navigation and content preview
- Visual content gallery
- Direct image viewing

## 📊 Performance Metrics

EduRAG provides comprehensive evaluation including:

| Category | Metric | Description |
|----------|--------|-------------|
| **Retrieval** | MRR, Recall@1, Recall@5 | How well sources are found |
| **Generation** | BERTScore, ROUGE-L, F1 | Answer quality and accuracy |
| **Grounding** | Faithfulness, Citation Validity | Trustworthiness measures |
| **Explainability** | Token Grounding Ratio | How much of answer is verifiable |

## 🔧 Configuration Options

### Model Selection
- Primary LLM: `deepseek-r1:8b` (high quality)
- Fallback LLM: `deepseek-r1:1.5b` (resource efficient)
- Vision Model: `llava-phi3` (image description)
- Embeddings: `BAAI/bge-small-en-v1.5`

### Adjustable Parameters
- Chunk size and overlap
- Retrieval depth (top-k)
- HyDE activation
- VLM processing for images

## 🛡️ Privacy & Security

- **Local Processing**: All computations happen on your machine
- **No Cloud Dependencies**: Except for optional Ollama models
- **Data Control**: Full ownership of your educational materials
- **Secure Storage**: Knowledge base stored locally in CSV format

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [Docling](https://ds4sd.github.io/docling/) for document parsing
- [Ollama](https://ollama.com/) for local LLM inference
- [LangChain](https://langchain.com/) for RAG framework
- [Streamlit](https://streamlit.io/) for web interface
- [Hugging Face](https://huggingface.co/) for embedding models

## 📞 Support

For issues, questions, or feature requests, please open an issue on GitHub.

---

*EduRAG transforms educational materials into an interactive, explorable knowledge base with state-of-the-art AI capabilities while maintaining complete privacy and control over your data.*
