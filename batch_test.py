import pandas as pd
import os
import json
import time
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ImportError:
    from langchain_community.chat_models import ChatOpenAI
    from langchain_community.embeddings import OpenAIEmbeddings

from rag import RAGSystem

# Set Environment Variables globally
os.environ["OPENAI_API_KEY"] = "sk-zk232b67a8a05e6026a6fcf53148dca1fc065a62d0885020"
os.environ["OPENAI_BASE_URL"] = "https://api.zhizengzeng.com/v1"

# Hit Rate 配置：允许的页码误差范围（±N 页）
PAGE_TOLERANCE = 1  # 0=严格匹配, 1=允许前后1页, 2=允许前后2页

def evaluate_rag(csv_path: str, output_path: str = "evaluation_results.json", config: dict = {}):
    # Load Data
    print(f"Loading test data from {csv_path}...")
    df = pd.read_csv(csv_path,encoding='gbk')
    
    # Verify columns
    required_columns = ['question', 'ground_truth', 'page', 'filename']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")

    # Initialize RAG
    print("Initializing RAG System...")
    rag = RAGSystem(config)
    print(rag.config)
    # Get experiment config
    experiment_config = rag.config.copy()
    
    # Import additional config from rag.py
    from rag import USE_VLLM, FAISS_DB_PATH, EMBED_MODEL_ID, LOCAL_LLM_PATH
    
    # Add system-level configuration
    experiment_config["system_config"] = {
        "use_vllm": USE_VLLM,
        "faiss_db_path": FAISS_DB_PATH,
        "embed_model_id": EMBED_MODEL_ID,
        "local_llm_path": LOCAL_LLM_PATH,
        "page_tolerance": PAGE_TOLERANCE,
    }
    
    # Inference
    print("Running Inference...")
    results = []
    inference_latencies = []
    
    for index, row in df.iterrows():
        q = row['question']
        gt = row['ground_truth']
        gt_page = str(row['page'])  # 页码字符串，如 "12,13"
        gt_filename = row['filename']
        
        print(f"Processing {index+1}/{len(df)}: {q}")
        
        # Call RAG
        response = rag.query(q)
        
        inference_latencies.append(response.get("latency_seconds", 0))
        
        # Extract contexts for Ragas (list of strings)
        ragas_contexts = [doc['content'] for doc in response['source_documents']]
        
        # 计算命中率
        gt_pages = set(int(p.strip()) for p in gt_page.split(',') if p.strip())
        
        def check_hit(docs, gt_pages, gt_filename):
            """检查是否命中正确的页码和文件（支持页码误差范围）"""
            # 扩展 ground truth 页码范围（±PAGE_TOLERANCE）
            expanded_gt_pages = set()
            for page in gt_pages:
                for offset in range(-PAGE_TOLERANCE, PAGE_TOLERANCE + 1):
                    if page + offset > 0:  # 页码必须为正数
                        expanded_gt_pages.add(page + offset)
            
            # 检查文档是否命中
            for doc in docs:
                doc_filename = doc.get('source', '')
                doc_pages = set(doc.get('page', []))
                # 文件名匹配 + 扩展后的页码有交集
                if gt_filename in doc_filename and expanded_gt_pages & doc_pages:
                    return True
            return False
        
        initial_hit = check_hit(response['retrieved_docs'], gt_pages, gt_filename)
        reranked_hit = check_hit(response['source_documents'], gt_pages, gt_filename)
        
        results.append({
            "question": q,
            "answer": response['answer'],
            "contexts": ragas_contexts,
            "ground_truth": gt,
            "ground_truth_page": gt_page,
            "ground_truth_filename": gt_filename,
            "latency": response.get("latency_seconds", 0),
            "retrieval_details": {
                "reranked_docs": response['source_documents'],
                "initial_retrieval_docs": response['retrieved_docs']
            },
            "hit_analysis": {
                "initial_retrieval_hit": initial_hit,
                "reranked_hit": reranked_hit
            }
        })

    # Prepare RAGAS Dataset
    # Note: Ragas expects specific column names
    MAX_CONTEXT_LENGTH = 1200
    MAX_CONTEXTS = 5
    ragas_df = pd.DataFrame([{
        "question": r["question"],
        "answer": r["answer"],
        "contexts": [ctx[:MAX_CONTEXT_LENGTH] for ctx in r["contexts"][:MAX_CONTEXTS]],
        "ground_truth": r["ground_truth"]
    } for r in results])
    
    dataset = Dataset.from_pandas(ragas_df)
    
    # Initialize LLM and Embeddings for RAGAS Evaluation
    print("Initializing Evaluator Models...")
    evaluator_llm = ChatOpenAI(
        model="gpt-5-nano",
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        temperature=0,
        max_tokens=8192,
        timeout=180,
        max_retries=3,
        n=3
    )
    evaluator_embeddings = OpenAIEmbeddings(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        timeout=120,
        max_retries=3
    )

    # Evaluate
    print("Running RAGAS Evaluation...")
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]
    
    scores = evaluate(
        dataset=dataset, 
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        batch_size=len(df) # Process all at once
    )
    
    # Calculate Aggregate Metrics
    scores_df = scores.to_pandas()
    avg_scores = scores_df.mean(numeric_only=True).to_dict()
    avg_latency = sum(inference_latencies) / len(inference_latencies) if inference_latencies else 0
    
    # Calculate Hit Rates
    initial_hits = sum(1 for r in results if r['hit_analysis']['initial_retrieval_hit'])
    reranked_hits = sum(1 for r in results if r['hit_analysis']['reranked_hit'])
    total_samples = len(results)
    
    initial_hit_rate = initial_hits / total_samples if total_samples > 0 else 0
    reranked_hit_rate = reranked_hits / total_samples if total_samples > 0 else 0
    
    # Construct Final JSON Output
    final_output = {
        "experiment_config": experiment_config,
        "summary_metrics": {
            "average_faithfulness": avg_scores.get("faithfulness", 0),
            "average_answer_relevancy": avg_scores.get("answer_relevancy", 0),
            "average_context_precision": avg_scores.get("context_precision", 0),
            "average_context_recall": avg_scores.get("context_recall", 0),
            "average_end_to_end_latency_seconds": avg_latency,
            "initial_retrieval_hit_rate": initial_hit_rate,
            "reranked_hit_rate": reranked_hit_rate,
            "total_samples": len(results)
        },
        "detailed_results": []
    }
    
    # Merge Ragas scores back into detailed results
    for i, res in enumerate(results):
        # Get scores for this sample
        sample_scores = scores_df.iloc[i].to_dict()
        
        # Filter out the input columns to keep just the metrics
        metric_scores = {k: v for k, v in sample_scores.items() 
                        if k in [m.name for m in metrics]}
        
        detail_entry = {
            "question": res["question"],
            "ground_truth": res["ground_truth"],
            "ground_truth_source": {
                "page": res["ground_truth_page"],
                "filename": res["ground_truth_filename"]
            },
            "generated_answer": res["answer"],
            "metrics": metric_scores,
            "latency_seconds": res["latency"],
            "hit_analysis": res["hit_analysis"],
            "retrieval_info": {
                "groundedness_check": {
                    "faithfulness_score": metric_scores.get("faithfulness"),
                    "citation_validity_check": "See retrieved_docs for source pages"
                },
                "retrieved_documents_reranked": res["retrieval_details"]["reranked_docs"],
                "retrieved_documents_initial": res["retrieval_details"]["initial_retrieval_docs"]
            }
        }
        final_output["detailed_results"].append(detail_entry)

    # Save Results
    print("Saving results...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
        
    print(f"\nEvaluation complete. Results saved to {output_path}")
    print(f"\n{'='*60}")
    print("Experiment Configuration:")
    print(f"{'='*60}")
    print(f"  Name: {experiment_config.get('name', 'N/A')}")
    print(f"  Top-K: {experiment_config.get('top_k', 'N/A')}")
    print(f"  Rerank Top-K: {experiment_config.get('rerank_top_k', 'N/A')}")
    print(f"\nRetrieval Enhancements:")
    print(f"  - Query Expansion:     {'✓' if experiment_config.get('use_query_expansion') else '✗'}", end="")
    if experiment_config.get('use_query_expansion'):
        print(f" (count: {experiment_config.get('expansion_count', 'N/A')})")
    else:
        print()
    print(f"  - HyDE:                {'✓' if experiment_config.get('use_hyde') else '✗'}")
    print(f"  - Doc Filter (NER):    {'✓' if experiment_config.get('use_doc_filter') else '✗'}")
    print(f"  - Refine Generation:   {'✓' if experiment_config.get('use_refine_generation') else '✗'}")
    
    # Display system configuration
    sys_config = experiment_config.get('system_config', {})
    if sys_config:
        print(f"\nSystem Configuration:")
        print(f"  - Use vLLM:            {'✓' if sys_config.get('use_vllm') else '✗'}")
        print(f"  - Vector DB:           {sys_config.get('faiss_db_path', 'N/A')}")
        print(f"  - Page Tolerance:      ±{sys_config.get('page_tolerance', 'N/A')} pages")
    
    print(f"\n{'='*60}")
    print("Summary Metrics:")
    print(f"{'='*60}")
    print(f"Ragas Scores:")
    print(f"  - Faithfulness:        {avg_scores.get('faithfulness', 0):.4f}")
    print(f"  - Answer Relevancy:    {avg_scores.get('answer_relevancy', 0):.4f}")
    print(f"  - Context Precision:   {avg_scores.get('context_precision', 0):.4f}")
    print(f"  - Context Recall:      {avg_scores.get('context_recall', 0):.4f}")
    print(f"\nRetrieval Metrics:")
    print(f"  - Initial Hit Rate:    {initial_hit_rate:.2%}")
    print(f"  - Reranked Hit Rate:   {reranked_hit_rate:.2%}")
    print(f"\nPerformance:")
    print(f"  - Avg Latency:         {avg_latency:.4f}s")
    print(f"  - Total Samples:       {total_samples}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--desc", type=str, default="", help="Description suffix for output filename")
    parser.add_argument("--config_file",type=str, default="./config_mrt.json")
    parser.add_argument("--exp_name",type=str, required=True,help="Select from ['experiment_0_baseline','experiment_1_query_expansion','experiment_2_hyde','experiment_3_doc_filter','experiment_4_expansion_plus_filter','experiment_5_refine_generation','experiment_6_full_pipeline']")
    args = parser.parse_args()
    
    # Generate output filename: output/test_YYYYMMDD_HHMMSS_desc.json
    import os
    os.makedirs("output", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    desc_suffix = f"_{args.exp_name}" if args.exp_name else "" # 修改成exp_name
    output_path = f"output/test_{desc_suffix}.json"
    import json
    with open(args.config_file, 'r', encoding='utf-8') as f:

        all_configs = json.load(f)
    config = all_configs.get(args.exp_name, {})
    evaluate_rag(args.file,output_path,config=config)
