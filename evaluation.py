"""
evaluation.py
Comprehensive benchmarking for Educational RAG
Connected directly to main_rag_withoutShap.py
"""
import pandas as pd
import numpy as np
import logging
import warnings
import evaluate
from typing import Dict, List
from tqdm import tqdm
from collections import defaultdict

from main_rag_withoutShap import get_hybrid_retriever_from_csv, run_rag_pipeline
from explainability import CitationGroundingMetrics

# Setup logging to keep the terminal clean during evaluation
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

class RAGEvaluator:
    def __init__(self, csv_path: str = "educational_knowledge_base.csv"):
        self.csv_path = csv_path
        
        logger.info("Loading NLP Evaluation Models (BERTScore, ROUGE)...")
        self.bertscore = evaluate.load("bertscore")
        self.rouge = evaluate.load("rouge")
        
        # Attribution tools
        self.citation_eval = CitationGroundingMetrics()
        
        # Initialize the hybrid retriever
        logger.info("Initializing Hybrid Retriever Engine...")
        self.retriever = get_hybrid_retriever_from_csv(self.csv_path, top_k=5, enable_hyde=False)
    
    # ============== TEXT METRICS ==============
    
    def exact_match(self, prediction: str, reference: str) -> float:
        """Strict exact string match."""
        return float(prediction.strip().lower() == reference.strip().lower())
    
    def f1_score(self, prediction: str, reference: str) -> float:
        """Token-level overlap F1 score."""
        pred_tokens = set(prediction.lower().split())
        ref_tokens = set(reference.lower().split())
        if not pred_tokens or not ref_tokens: 
            return 0.0
        
        common = pred_tokens & ref_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        
        if precision + recall == 0: 
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    # ============== SINGLE EVALUATION ==============
    
    def evaluate_single(self, query: str, ground_truth: str, target_chunk_id: str) -> Dict:
        """Runs a single query through the RAG and computes all metrics."""
        # 1. Execute RAG
        result = run_rag_pipeline(query, self.retriever, llm=None)
        
        prediction = result.answer
        context_docs = result.context
        retrieved_ids = [d.metadata.get("chunk_id") for d in context_docs]
        
        # 2. Generation Metrics (Accuracy)
        gen_metrics = {
            "exact_match": self.exact_match(prediction, ground_truth),
            "f1": self.f1_score(prediction, ground_truth),
            "rouge_l": self.rouge.compute(predictions=[prediction], references=[ground_truth])['rougeL'],
            "bertscore_f1": np.mean(self.bertscore.compute(predictions=[prediction], references=[ground_truth], lang="en")['f1'])
        }
        
        # 3. Retrieval Metrics (Vector Math)
        # Did we find the exact chunk the "Teacher" used to write the question?
        r_at_1 = 1 if len(retrieved_ids) > 0 and retrieved_ids[0] == target_chunk_id else 0
        r_at_5 = 1 if target_chunk_id in retrieved_ids[:5] else 0
        
        mrr = 0.0
        if target_chunk_id in retrieved_ids:
            mrr = 1.0 / (retrieved_ids.index(target_chunk_id) + 1)
            
        ret_metrics = {
            "recall@1": r_at_1,
            "recall@5": r_at_5,
            "mrr": mrr
        }
        
        # 4. Hallucination & Citation Metrics
        cit_metrics = self.citation_eval.evaluate_citations(prediction, context_docs)
        
        return {
            "generation": gen_metrics,
            "retrieval": ret_metrics,
            "grounding": {
                "citation_validity": cit_metrics.get("valid_citation_rate", 0),
                "faithfulness": result.faithfulness,
                "token_grounding_ratio": result.token_attribution.get("grounding_ratio", 0)
            }
        }
    

    # ============== BATCH EVALUATION ==============
    
    def evaluate_dataset(self, eval_csv: str = "evaluation_dataset.csv") -> Dict:
        """Processes the entire Ground Truth dataset."""
        try:
            df = pd.read_csv(eval_csv)
        except Exception as e:
            logger.error(f"Failed to load evaluation dataset: {e}")
            return {}
        
        logger.info(f"\n🚀 Beginning rigorous academic benchmark on {len(df)} queries...")
        all_results = defaultdict(list)
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating RAG System"):
            try:
                result = self.evaluate_single(
                    query=str(row["query"]), 
                    ground_truth=str(row["answer"]), 
                    target_chunk_id=str(row["target_chunk_id"]).strip()
                )
                
                # Flatten the dictionaries to calculate means easily
                for category, metrics in result.items():
                    for k, v in metrics.items():
                        all_results[f"{category}_{k}"].append(v)
                        
            except Exception as e:
                logger.error(f"Error evaluating query {idx}: {e}")
                continue
        
        # Calculate final mathematical averages
        summary = {k: np.mean(v) for k, v in all_results.items() if v}
        
        # Print the formal report
        self._print_report(summary, len(df))
        
        # Save raw results for the paper
        output_file = "final_benchmark_results.json"
        import json
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=4)
        logger.info(f"\n✅ Detailed metrics saved to {output_file}")
        
        return summary
    
    def _print_report(self, summary: Dict, total: int):
        print("\n" + "="*65)
        print("                 RESEARCH BENCHMARK REPORT                 ")
        print("="*65)
        print(f"Total Exam Queries Evaluated: {total}")
        print("-" * 65)
        print("1. RETRIEVAL ENGINE PERFORMANCE (Hybrid RRF)")
        print(f"   MRR (Mean Reciprocal Rank): {summary.get('retrieval_mrr', 0):.4f} (Higher is better)")
        print(f"   Recall@1:                   {summary.get('retrieval_recall@1', 0):.4f}")
        print(f"   Recall@5:                   {summary.get('retrieval_recall@5', 0):.4f}")
        print("-" * 65)
        print("2. GENERATION ACCURACY (DeepSeek R1)")
        print(f"   BERTScore F1 (Semantic):    {summary.get('generation_bertscore_f1', 0):.4f}")
        print(f"   ROUGE-L (Structure):        {summary.get('generation_rouge_l', 0):.4f}")
        print(f"   Exact F1 (Token Overlap):   {summary.get('generation_f1', 0):.4f}")
        print("-" * 65)
        print("3. TRUST & HALLUCINATION SAFEGUARDS")
        print(f"   Faithfulness Score:         {summary.get('grounding_faithfulness', 0):.4f}")
        print(f"   Valid Citation Rate:        {summary.get('grounding_citation_validity', 0)*100:.1f}%")
        print(f"   Token Grounding Ratio:      {summary.get('grounding_token_grounding_ratio', 0)*100:.1f}%")
        print("="*65)

if __name__ == "__main__":
    evaluator = RAGEvaluator()
    results = evaluator.evaluate_dataset("evaluation_dataset.csv")
