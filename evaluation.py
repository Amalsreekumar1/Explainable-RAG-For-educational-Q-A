import pandas as pd
import numpy as np
import logging
import warnings
import evaluate
from tqdm import tqdm
from main_rag_app import get_hybrid_retriever_from_csv, get_qa_chain

# 1. Suppress Noisy SHAP & Library Logs
logging.getLogger("shap").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Setup simple logging for the evaluation script
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def evaluate_rag():
    # 2. Configuration
    CSV_PATH = "educational_knowledge_base.csv"  # Your main knowledge base
    EVAL_DATASET = "evaluation_dataset.csv"      # Your QA pairs for testing
    
    # Load Metrics
    bertscore = evaluate.load("bertscore")
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")

    # 3. Initialize RAG System
    logger.info("--- Initializing RAG System ---")
    retriever = get_hybrid_retriever_from_csv(CSV_PATH, top_k=5)
    qa_chain = get_qa_chain(retriever)

    # 4. Load Evaluation Data (FIXED COLUMN NAMES)
    try:
        df = pd.read_csv(EVAL_DATASET)
        # Check for your specific columns: 'query' and 'answer'
        if "query" not in df.columns or "answer" not in df.columns:
            raise ValueError("CSV must have 'query' and 'answer' columns.")
    except Exception as e:
        logger.error(f"Failed to load evaluation dataset: {e}")
        return

    # Map your CSV columns to the script variables
    questions = df["query"].tolist()
    ground_truths = df["answer"].tolist()
    
    predictions = []
    retrieval_scores = []

    logger.info(f"--- Starting Evaluation on {len(questions)} Questions ---")

    # 5. Evaluation Loop
    for q in tqdm(questions, desc="Processing Queries"):
        try:
            # Invoke RAG Chain
            response = qa_chain.invoke({"input": q})
            
            # Handle response structure
            ans = response["answer"] if isinstance(response, dict) else response
            predictions.append(ans)

            # Capture Retrieval Score directly from the retriever instance
            # The HybridRetriever class stores the score of the last query in self.last_retrieval_score
            score = getattr(retriever, "last_retrieval_score", 0.0)
            retrieval_scores.append(score)

        except Exception as e:
            logger.error(f"Error processing query '{q}': {e}")
            predictions.append("Error")
            retrieval_scores.append(0.0)

    # 6. Calculate Generation Metrics
    logger.info("--- Computing NLP Metrics ---")
    
    # BERTScore (Faithfulness/Similarity)
    bert_results = bertscore.compute(predictions=predictions, references=ground_truths, lang="en")
    avg_f1 = np.mean(bert_results['f1'])

    # ROUGE (Text Overlap)
    rouge_results = rouge.compute(predictions=predictions, references=ground_truths)
    
    # BLEU (Precision of n-grams)
    try:
        bleu_results = bleu.compute(predictions=predictions, references=ground_truths)
        bleu_score = bleu_results['bleu']
    except:
        bleu_score = 0.0

    # 7. Final Summary Output
    avg_retrieval = np.mean(retrieval_scores)
    
    print("\n" + "="*40)
    print("       RAG SYSTEM EVALUATION REPORT       ")
    print("="*40)
    print(f"Total Questions Evaluated: {len(questions)}")
    print("-" * 40)
    print(f"1. RETRIEVAL PERFORMANCE")
    print(f"   Average Retrieval Score:   {avg_retrieval:.4f} (Cosine Similarity)")
    print("-" * 40)
    print(f"2. GENERATION PERFORMANCE")
    print(f"   BERTScore F1 (Semantic):   {avg_f1:.4f}  <-- Primary Metric")
    print(f"   ROUGE-L (Overlap):         {rouge_results['rougeL']:.4f}")
    print(f"   BLEU Score:                {bleu_score:.4f}")
    print("-" * 40)
    print(f"3. OVERALL INTERPRETATION")
    if avg_f1 > 0.8:
        print("   ✅ EXCELLENT: High semantic accuracy.")
    elif avg_f1 > 0.6:
        print("   ⚠️ GOOD: Mostly accurate, but check low-scoring queries.")
    else:
        print("   ❌ NEEDS IMPROVEMENT: Low semantic matching.")
    print("="*40)

    # Optional: Save detailed results to CSV
    df['generated_answer'] = predictions
    df['retrieval_score'] = retrieval_scores
    df['bert_f1'] = bert_results['f1']
    df.to_csv("evaluation_results_detailed.csv", index=False)
    logger.info("\nDetailed results saved to 'evaluation_results_detailed.csv'")

if __name__ == "__main__":
    evaluate_rag()