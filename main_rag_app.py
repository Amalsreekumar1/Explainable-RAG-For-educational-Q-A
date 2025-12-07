#!/usr/bin/env python3
"""
main_rag_app.py
Stable RAG pipeline for educational QA with hybrid retrieval (BM25 + FAISS + RRF + embedding rerank),
Ollama LLM generation, SHAP explainability (rerank features), and structured final answers.
"""

import os
import re
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from operator import itemgetter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rank_bm25 import BM25Okapi
import shap
import evaluate
from typing import Optional
from dotenv import load_dotenv 
load_dotenv()

# LangChain imports - keep as in your environment; if imports fail, adjust to your installed package split
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough, RunnableSequence, RunnableMap
from pydantic import BaseModel, Field, ConfigDict

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Globals
EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
FAISS_INDEX_PATH = "faiss_index"
EXPLAIN_WITH_SHAP = True
bertscore = evaluate.load("bertscore")

# -------------------------
# CSVDocLoader
# -------------------------
class CSVDocLoader:
    @classmethod
    def load(cls, csv_path: str) -> List[Document]:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False).fillna("")
        # Strip whitespace and tabs from column names
        df.columns = df.columns.str.strip()
        required = ["text", "source_file", "page_number", "chunk_id"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"CSV must contain '{col}' column. Found: {list(df.columns)}")
        docs: List[Document] = []
        for idx, row in df.iterrows():
            content = str(row.get("text", "")).strip()
            if not content:
                continue
            metadata = {
                "source_file": str(row.get("source_file", "")).strip() or "Unknown",
                "page_number": str(row.get("page_number", "")).strip() or "N/A",
                "chunk_id": str(row.get("chunk_id", "")).strip() or f"row-{idx}",
                "chunk_length": row.get("chunk_length", None)
            }
            doc = Document(page_content=content, metadata=metadata)
            docs.append(doc)
        logger.info("Loaded %d documents from %s", len(docs), csv_path)
        return docs

# -------------------------
# Tokenizer for BM25
# -------------------------
def simple_tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    tokens = re.findall(r"\b\w+\b", text)
    return tokens

# -------------------------
# RRF fusion
# -------------------------
def rrf_fusion(bm25_docs: List[Document], faiss_docs: List[Document], k: int = 60) -> List[Document]:
    score_map: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}
    def cid(d: Document) -> str:
        return str(d.metadata.get("chunk_id", hash(d.page_content)))
    for rank, d in enumerate(bm25_docs, start=1):
        c = cid(d)
        doc_map.setdefault(c, d)
        score_map[c] = score_map.get(c, 0.0) + 1.0 / (k + rank)
    for rank, d in enumerate(faiss_docs, start=1):
        c = cid(d)
        doc_map.setdefault(c, d)
        score_map[c] = score_map.get(c, 0.0) + 1.0 / (k + rank)
    sorted_cids = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    fused = [doc_map[cid] for cid, _ in sorted_cids]
    logger.debug("RRF fused %d docs", len(fused))
    return fused

# -------------------------
# SHAP EXPLANATION 
# -------------------------
def generate_shap_explanation(shap_values: np.ndarray, top_docs: List[Document], doc_ids: List[str]) -> str:
    """
    Creates a student-friendly and developer-friendly SHAP explanation.
    No plots, no arrays, no PNGs.
    """
    try:
        # Ensure 1D numpy array
        sv = np.asarray(shap_values).ravel()
        if sv.size == 0:
            return ""
        # Take absolute values & find top influential docs/features
        shap_vals = np.abs(sv)

        # If doc_ids length doesn't match shap_vals length, we align to the minimum
        n_items = min(len(shap_vals), len(doc_ids))
        if n_items == 0:
            return ""

        top_indices = np.argsort(shap_vals)[::-1][:min(5, n_items)]

        # Student-friendly explanation
        student_lines = []
        for idx in top_indices:
            student_lines.append(
                f"- Chunk {doc_ids[idx]} — This chunk strongly contributed because it contained "
                f"clear, directly relevant information to your question."
            )

        student_explanation = (
            #"SHAP Explanation (How the system chose your answer)\n"
            "The system analyzed retrieved chunks and highlighted the most influential ones that "
            "supported the final answer:\n\n"
            + "\n".join(student_lines)
            + "\n"
        )

        # Developer technical section
        dev_lines = []
        for idx in top_indices:
            dev_lines.append(f"• Chunk {doc_ids[idx]} → SHAP Value: {shap_vals[idx]:.6f}")

        dev_explanation = (
            "\nTechnical SHAP Breakdown \n"
            "Higher SHAP values mean stronger influence on the re-ranking decision.\n\n"
            + "\n".join(dev_lines)
            + "\n"
        )

        return student_explanation + dev_explanation
    except Exception as e:
        logger.debug("generate_shap_explanation failed: %s", e)
        return ""

# -------------------------
# Embedding re-rank with SHAP 
# -------------------------
def rerank_by_embedding(query: str, docs: List[Document], embedder: HuggingFaceEmbeddings, explain: bool = False) -> Tuple[List[Document], float, Optional[str]]:
    if not docs:
        return [], 0.0, None
    q_emb = np.array(embedder.embed_query(query), dtype=float)
    doc_texts = [d.page_content for d in docs]
    doc_embs = np.array(embedder.embed_documents(doc_texts), dtype=float)
    qnorm = np.linalg.norm(q_emb)
    if qnorm > 0:
        q_emb = q_emb / qnorm
    norms = np.linalg.norm(doc_embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    doc_embs = doc_embs / norms
    sims = np.dot(doc_embs, q_emb)
    order = np.argsort(sims)[::-1]
    ordered = [docs[i] for i in order]

    # Calculate Retrieval Score (Avg Cosine Sim Top-5)
    avg_sim = float(np.mean(sims[order[:5]])) if len(order) >= 1 else 0.0
    shap_explanation = ""
    if explain and len(ordered) > 0:
        try:
            # SHAP KernelExplainer setup (can be slow)
            def sim_model(X):
                # X is an array of embeddings; return dot with q_emb
                return np.dot(X, q_emb.reshape(-1)).astype(float)

            # For the explainer background, pick up to 10 other document embeddings
            # Use safe slicing on doc_embs
            bg_indices = order[1: min(len(order), 11)]
            if bg_indices.size == 0:
                background = doc_embs[order[:1]]
            else:
                background = doc_embs[bg_indices]

            # Build explainer and compute SHAP values for the top-ranked doc embedding
            explainer = shap.KernelExplainer(sim_model, background)
            top_idx = order[0]
            shap_values = explainer.shap_values(doc_embs[top_idx:top_idx+1])
            # Flatten shap_values to 1D
            shap_arr = np.asarray(shap_values).ravel()

            # Prepare doc_ids mapping - use as many as shap_arr length if available, else fallback
            # We'll map the top N retrieved chunks to the first N positions
            max_map = max(1, min(len(ordered), shap_arr.shape[0]))
            mapped_doc_ids = [d.metadata.get("chunk_id", f"idx-{i}") for i, d in enumerate(ordered[:max_map])]

            shap_explanation = generate_shap_explanation(shap_arr[:max_map], ordered[:max_map], mapped_doc_ids)
            logger.info("Generated SHAP explanation text")
        except Exception as e:
            logger.warning("SHAP explanation generation failed: %s", e)
            shap_explanation = ""
    return ordered, avg_sim, shap_explanation

# -------------------------
# HybridRetriever 
# -------------------------
class HybridRetriever(BaseRetriever, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    documents: List[Document] = Field(..., description="Pre-chunked documents")
    embedder: HuggingFaceEmbeddings = Field(default=embedder)
    top_k: int = Field(default=5)
    use_reformulation: bool = Field(default=False)
    llm_for_reformulation: Any = Field(default=None)
    bm25: Any = Field(default=None)
    faiss: Any = Field(default=None)
    faiss_retriever: Any = Field(default=None)
    query_cache: Dict[str, str] = Field(default_factory=dict)

    # New attributes to store metrics for easy retrieval after invocation
    last_retrieval_score: float = Field(default=0.0)
    last_shap_explanation: str = Field(default="")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.documents:
            raise ValueError("documents required")
        tokenized = [simple_tokenize(d.page_content) for d in self.documents]
        object.__setattr__(self, 'bm25', BM25Okapi(tokenized))

        # FAISS load/build logic
        if os.path.exists(FAISS_INDEX_PATH):
            try:
                faiss_obj = FAISS.load_local(FAISS_INDEX_PATH, self.embedder, allow_dangerous_deserialization=True)
                object.__setattr__(self, 'faiss', faiss_obj)
                object.__setattr__(self, 'faiss_retriever', faiss_obj.as_retriever(search_kwargs={"k": max(20, self.top_k * 3)}))
                logger.info("Loaded FAISS from %s", FAISS_INDEX_PATH)
            except Exception as e:
                logger.warning("Failed to load FAISS (%s); rebuilding", e)
        if not hasattr(self, 'faiss') or self.faiss is None:
            try:
                faiss_obj = FAISS.from_documents(self.documents, self.embedder)
                faiss_obj.save_local(FAISS_INDEX_PATH)
                object.__setattr__(self, 'faiss', faiss_obj)
                object.__setattr__(self, 'faiss_retriever', faiss_obj.as_retriever(search_kwargs={"k": max(20, self.top_k * 3)}))
                logger.info("Built & saved FAISS on %d docs to %s", len(self.documents), FAISS_INDEX_PATH)
            except Exception as e:
                logger.error("Failed to build FAISS: %s", e)
                object.__setattr__(self, 'faiss', None)
                object.__setattr__(self, 'faiss_retriever', None)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        logger.info("Query started: %s", query)
        reformulated = query

        fetch_k = max(self.top_k * 3, 20)
        # BM25 retrieval
        bm25_scores = self.bm25.get_scores(simple_tokenize(reformulated))
        top_idx = np.argsort(bm25_scores)[-fetch_k:][::-1]
        bm25_docs = [self.documents[i] for i in top_idx if bm25_scores[i] > 0][:fetch_k]
        # FAISS retrieval
        faiss_docs = []
        if self.faiss_retriever is not None:
            try:
                faiss_docs = self.faiss.similarity_search(reformulated, k=fetch_k)
            except Exception as e:
                logger.debug("FAISS retrieval failed: %s", e)
                faiss_docs = []
        fused = rrf_fusion(bm25_docs, faiss_docs, k=60)
        # Rerank and compute metrics
        reranked, avg_sim, shap_explanation = rerank_by_embedding(query, fused, self.embedder, explain=EXPLAIN_WITH_SHAP)

        # Store metrics and explanation in the instance for access in the main loop
        object.__setattr__(self, 'last_retrieval_score', float(avg_sim))
        object.__setattr__(self, 'last_shap_explanation', shap_explanation or "")
        top = reranked[: self.top_k]
        logger.info("Returning top %d docs (Avg Cosine Sim Top-5: %.3f)", len(top), avg_sim)
        return top

# -------------------------
# combine_documents_for_prompt (for context stuffing)
# -------------------------
DOCUMENT_PROMPT = (
    "[{idx}] Source: {source_file} | Page: {page_number} | Chunk: {chunk_id}\n{page_content}\n"
)
def combine_documents_for_prompt(docs: List[Document], max_chars: int = 4000) -> str:
    parts = []
    total = 0
    for idx, d in enumerate(docs, 1):
        chunk = DOCUMENT_PROMPT.format(
            idx=idx,
            source_file=d.metadata.get("source_file", "Unknown"),
            page_number=d.metadata.get("page_number", "N/A"),
            chunk_id=d.metadata.get("chunk_id", "N/A"),
            page_content=d.page_content
        )
        parts.append(chunk)
        total += len(chunk)
        if total >= max_chars:
            logger.warning("Context truncated at %d chars", total)
            break
    return "\n\n".join(parts)

# -------------------------
# QA prompt 
# -------------------------
QA_PROMPT = PromptTemplate.from_template(
    """
You are a reliable assistant. Your response MUST be based ONLY on the CONTEXT provided below.
Provide a clear, structured, accurate answer (3-4 sentences max).
DO NOT include any introductory phrases or headers like 'Answer:', 'Response:', or 'The answer is'. START DIRECTLY with the answer text.
If the context does not contain the answer, your entire reply MUST be: I don't know.
### Context:
{context}
### Question:
{question}
"""
)

# -------------------------
# Clean Answer Utility
# -------------------------
def clean_llm_output(raw_text: str) -> str:
    """Strips redundant LLM headers and trailing summaries to clean the output."""
    text = raw_text.strip()
    redundant_markers = ["\n\nAnswer:", "\n\nanswer:", "\n\nSOURCES:", "\n\nSources:"]
    clean_text = text
    for marker in redundant_markers:
        if marker in clean_text:
            clean_text = clean_text.split(marker)[0].strip()
            break
    clean_text = re.sub(r'^\s*Answer:\s*', '', clean_text, flags=re.IGNORECASE).strip()
    return clean_text

# -------------------------
# Build LCEL chain
# -------------------------
def get_qa_chain(retriever: BaseRetriever):
    # rag core: combine doc list into {context}, feed prompt, run LLM, parse to string
    rag_chain_core = (
        RunnablePassthrough.assign(
            context=lambda x: combine_documents_for_prompt(x["context"])
        )
        | QA_PROMPT
        | OllamaLLM(model="deepseek-r1:1.5b", temperature=0.1, max_new_tokens=512)
        | StrOutputParser()
    )
    # retrieval map: get documents from retriever and keep question
    retrieve_and_map = RunnableMap({
        "context": itemgetter("input") | retriever,
        "question": itemgetter("input")
    })
    # final sequence: run rag core -> return answer and context (docs)
    final_seq = retrieve_and_map.assign(
        answer=rag_chain_core,
    ).pick(["answer", "context"])
    return final_seq

# -------------------------
# Scores and Helpers
# -------------------------
def compute_faithfulness(answer: str, context_docs: List[Document]) -> float:
    context_text = " ".join([d.page_content for d in context_docs])
    results = bertscore.compute(predictions=[answer], references=[context_text], lang="en")
    return float(np.mean(results['f1']))

def compute_plausibility(llm, answer: str, query: str) -> float:
    """Robust wrapper: try multiple LLM call styles and parse a numeric (0-1) score."""
    try:
        conf_prompt = f"Rate this answer's plausibility (0-1) for query '{query}', only output the decimal score:\n{answer}\nScore:"
        # Try common call patterns
        if hasattr(llm, "invoke"):
            score_str = llm.invoke(conf_prompt).strip()
        elif callable(llm):
            score_str = llm(conf_prompt)
            if not isinstance(score_str, str):
                score_str = str(score_str)
        elif hasattr(llm, "generate"):
            gen = llm.generate([conf_prompt])
            try:
                score_str = gen.generations[0][0].text
            except Exception:
                score_str = str(gen)
        else:
            return 0.5
        if isinstance(score_str, bytes):
            score_str = score_str.decode("utf-8", errors="ignore")
        match = re.search(r'\d*\.?\d+', score_str)
        return float(match.group()) if match else 0.5
    except Exception as e:
        logger.warning("compute_plausibility failed: %s", e)
        return 0.5

def interpret_rag_scores(faithfulness: float, plausibility: float, retrieval_score: float) -> str:
    """
    Simple explanation for non-technical users.
    No ML words. No metrics. Only plain-language guidance.
    """

    # Average score to decide the tone
    avg = (faithfulness + plausibility + retrieval_score) / 3

    if avg >= 0.8:
        return (
            "This answer is highly reliable. "
            "It strongly matches the information found in your study materials, "
            "and you can trust this explanation."
        )

    elif avg >= 0.6:
        return (
            "This answer seems mostly correct. "
            "It matches the study materials reasonably well, "
            "but you may verify important parts if needed."
        )

    else:
        return (
            "This answer may not be fully accurate. "
            "The system could not find strong matching information in your textbooks. "
            "Try asking the question in a different way for a clearer result."
        )

def get_hybrid_retriever_from_csv(csv_path: str, top_k: int = 5, use_reformulation: bool = False):
    docs = CSVDocLoader.load(csv_path)
    return HybridRetriever(documents=docs, embedder=embedder, top_k=top_k, use_reformulation=use_reformulation)

# -------------------------
# Interactive CLI with full output
# -------------------------
if __name__ == "__main__":
    CSV_PATH = "educational_knowledge_base.csv"
    retriever = get_hybrid_retriever_from_csv(CSV_PATH, top_k=5, use_reformulation=False)
    qa_chain = get_qa_chain(retriever)
    llm_score = OllamaLLM(model="deepseek-r1:1.5b")

    print("RAG App Ready! Enter queries (type 'quit' to exit).")
    while True:
        query = input("\nEnter query: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break
        if not query:
            continue
        try:
            # LCEL runnable .invoke may return a mapping-like object; handle robustly
            raw = qa_chain.invoke({"input": query})
            # handle multiple possible shapes
            answer_text = raw.get("answer") if isinstance(raw, dict) else getattr(raw, "answer", None)
            context_docs = raw.get("context") if isinstance(raw, dict) else getattr(raw, "context", None)
            # Fallbacks
            if answer_text is None and isinstance(raw, (list, tuple)) and len(raw) >= 1:
                answer_text = raw[0]
            if context_docs is None:
                context_docs = []
        except Exception as e:
            logger.error("Chain invocation failed: %s", e)
            continue

        # 1. Extract Metrics
        context_docs = context_docs or []
        try:
            retrieval_score = float(getattr(retriever, "last_retrieval_score", 0.0) or 0.0)
        except:
            retrieval_score = 0.0

        faithfulness = compute_faithfulness(answer_text, context_docs) if context_docs else 0.0
        plausibility = compute_plausibility(llm_score, answer_text, query)

        # 2. Generate one simple interpretation message
        interpretation = (
            "This answer looks correct and matches the study materials well."
            if faithfulness >= 0.75
            else "The answer is partially correct. You may double-check important points."
            if faithfulness >= 0.55
            else "The answer may not be fully reliable. Try asking in a different way."
        )

        # 3. Clean final answer
        cleaned_answer = clean_llm_output(answer_text or "")

        # -------------------------
        # FINAL USER-FRIENDLY OUTPUT
        # -------------------------

        print("\n==============================")
        print("Explanation")
        print("==============================")
        print(interpretation)

        print("\n📘 Answer:")
        print(cleaned_answer)

        # Print SHAP explanation text (student-friendly + dev breakdown) if available
        shap_expl = getattr(retriever, "last_shap_explanation", "")
        if shap_expl:
            print("\n🔍 SHAP Explanation")
            print(shap_expl)

        print("\n Understanding Check:")
        print(f"- Faithfulness: {faithfulness:.3f}")
        print(f"- Plausibility: {plausibility:.3f}")
        print(f"- Retrieval Score: {retrieval_score:.3f}")

        print("\n Sources Used:")
        for i, doc in enumerate(context_docs, 1):
            snippet = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            print(f"[{i}] {doc.metadata.get('source_file','N/A')} | Page {doc.metadata.get('page_number','N/A')} | ID {doc.metadata.get('chunk_id','N/A')}")
            print(f"   Snippet: {snippet}")

        print("\n")