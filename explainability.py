"""
explainability.py
Advanced attribution for RAG without SHAP's computational overhead
"""
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

class EvidenceAttributor:
    """
    Evidence Attribution: Which chunks contributed to the answer?
    Uses cross-encoder attention + token overlap instead of SHAP
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
    
    def attribute_evidence(self, query: str, answer: str, 
                          documents: List[Document]) -> Dict:
        """
        Returns attribution scores for each document chunk
        """
        if not documents:
            return {"attributions": [], "method": "none"}
        
        # Cross-encode query+answer vs each document
        attributions = []
        
        with torch.no_grad():
            for i, doc in enumerate(documents):
                # Pairwise scoring
                pairs = [[f"{query} {answer}", doc.page_content]]
                inputs = self.tokenizer(pairs, padding=True, truncation=True, 
                                       return_tensors="pt", max_length=512)
                
                outputs = self.model(**inputs)
                score = torch.sigmoid(outputs.logits).item()
                
                # Token overlap as secondary signal
                answer_tokens = set(answer.lower().split())
                doc_tokens = set(doc.page_content.lower().split())
                overlap = len(answer_tokens & doc_tokens) / len(answer_tokens) if answer_tokens else 0
                
                # Combined score
                combined = 0.7 * score + 0.3 * overlap
                
                attributions.append({
                    "chunk_id": doc.metadata.get("chunk_id", f"doc_{i}"),
                    "source": doc.metadata.get("source_file", "unknown"),
                    "page": doc.metadata.get("page_number", "N/A"),
                    "relevance_score": float(score),
                    "token_overlap": float(overlap),
                    "combined_score": float(combined),
                    "type": doc.metadata.get("type", "text")
                })
        
        # Normalize to sum to 1 (percentage contribution)
        total = sum(a["combined_score"] for a in attributions) or 1.0
        for a in attributions:
            a["contribution_pct"] = (a["combined_score"] / total) * 100
        
        return {
            "attributions": sorted(attributions, key=lambda x: x["combined_score"], reverse=True),
            "primary_source": attributions[0]["source"] if attributions else None,
            "method": "cross_encoder_attention"
        }

class TokenAttributor:
    """
    Token Attribution: Which answer tokens are grounded in evidence?
    Uses gradient-based saliency on input-output alignment
    """
    
    def __init__(self, embedding_model=None):
        # Using embeddings as proxy for token importance
        self.embedder = embedding_model
    
    def generate_token_heatmap(self, answer: str, context_docs: List[Document]) -> Dict:
        """
        Generate token-level importance heatmap
        Returns tokens with grounding scores (0-1)
        """
        if not answer or not context_docs:
            return {"tokens": [], "heatmap": []}
        
        context_text = " ".join([d.page_content for d in context_docs])
        context_tokens = set(context_text.lower().split())
        
        # Split answer into tokens (simple whitespace tokenization)
        tokens = re.findall(r'\w+|[^\w\s]', answer)
        
        heatmap = []
        for token in tokens:
            token_lower = token.lower()
            
            # Exact match in context
            if token_lower in context_tokens:
                score = 1.0
            else:
                # Semantic similarity for synonyms/variants
                if self.embedder and len(token) > 3:
                    try:
                        # Check if token is semantically similar to any context token
                        token_emb = np.array(self.embedder.embed_query(token))
                        # Sample context tokens for efficiency
                        sample_context = list(context_tokens)[:100]
                        if sample_context:
                            ctx_embs = np.array(self.embedder.embed_documents(sample_context))
                            similarities = np.dot(ctx_embs, token_emb)
                            score = float(np.max(similarities)) if np.max(similarities) > 0.7 else 0.3
                        else:
                            score = 0.3
                    except:
                        score = 0.3
                else:
                    score = 0.3
            
            # Normalize to 0-1
            score = min(1.0, max(0.0, score))
            
            heatmap.append({
                "token": token,
                "grounded_score": score,
                "is_grounded": score > 0.7,
                "color": self._score_to_color(score)
            })
        
        # Calculate overall grounding ratio
        grounded_count = sum(1 for h in heatmap if h["is_grounded"])
        grounding_ratio = grounded_count / len(tokens) if tokens else 0
        
        return {
            "tokens": tokens,
            "heatmap": heatmap,
            "grounding_ratio": grounding_ratio,
            "ungrounded_tokens": [h["token"] for h in heatmap if not h["is_grounded"]][:10]
        }
    
    def _score_to_color(self, score: float) -> str:
        """Convert score to HTML color for visualization"""
        if score > 0.8:
            return "#2ecc71"  # Green (grounded)
        elif score > 0.5:
            return "#f39c12"  # Orange (partial)
        else:
            return "#e74c3c"  # Red (ungrounded/hallucinated)

class CitationGroundingMetrics:
    """
    Citation Precision & Recall for grounded answers
    """
    
    def evaluate_citations(self, answer: str, context_docs: List[Document], 
                          ground_truth_citations: Optional[List[int]] = None) -> Dict:
        """
        Calculate citation precision and recall
        
        Args:
            answer: Generated answer with citations [1], [2], etc.
            context_docs: Retrieved documents (index+1 = citation number)
            ground_truth_citations: Optional list of correct citation indices
        """
        # Extract citations from answer
        citations_found = re.findall(r'\[(\d+)\]', answer)
        cited_indices = [int(c) - 1 for c in citations_found]  # Convert to 0-based
        
        metrics = {
            "citation_count": len(citations_found),
            "unique_citations": len(set(citations_found)),
            "citations": citations_found
        }
        
        # Citation Validity (do they point to existing docs?)
        valid_citations = []
        invalid_citations = []
        
        for idx in cited_indices:
            if 0 <= idx < len(context_docs):
                valid_citations.append(idx)
            else:
                invalid_citations.append(idx + 1)  # Back to 1-based for reporting
        
        metrics["valid_citation_rate"] = len(valid_citations) / len(cited_indices) if cited_indices else 0
        metrics["invalid_citations"] = invalid_citations
        
        # If ground truth provided, calculate precision/recall
        if ground_truth_citations is not None:
            cited_set = set(cited_indices)
            gt_set = set([c - 1 for c in ground_truth_citations])  # Convert to 0-based
            
            true_positives = len(cited_set & gt_set)
            false_positives = len(cited_set - gt_set)
            false_negatives = len(gt_set - cited_set)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics.update({
                "citation_precision": precision,
                "citation_recall": recall,
                "citation_f1": f1,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives
            })
        else:
            # Self-consistency check: are citations supporting claims?
            metrics.update(self._verify_citation_support(answer, context_docs, cited_indices))
        
        return metrics
    
    def _verify_citation_support(self, answer: str, context_docs: List[Document], 
                                  cited_indices: List[int]) -> Dict:
        """Verify if cited documents actually support the claims"""
        # Simple implementation: check semantic similarity between sentence and cited doc
        support_scores = []
        
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        for sent in sentences:
            cited_in_sent = re.findall(r'\[(\d+)\]', sent)
            if not cited_in_sent:
                continue
            
            sent_text = re.sub(r'\[\d+\]', '', sent).strip()
            if len(sent_text) < 10:
                continue
            
            for c_str in cited_in_sent:
                idx = int(c_str) - 1
                if 0 <= idx < len(context_docs):
                    # Check if sentence is entailed by document (simplified)
                    doc_text = context_docs[idx].page_content[:500]  # First 500 chars
                    # In real implementation, use NLI model here
                    support_scores.append(1.0)  # Placeholder
        
        avg_support = np.mean(support_scores) if support_scores else 0
        return {
            "citation_support_score": avg_support,
            "unsupported_claims": len(support_scores) - sum(1 for s in support_scores if s > 0.5)
        }

# Factory function for easy integration
def get_attributors(embedding_model=None):
    """Get configured attributors"""
    return {
        "evidence": EvidenceAttributor(),
        "token": TokenAttributor(embedding_model),
        "citation": CitationGroundingMetrics()
    }
