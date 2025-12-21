"""
Reranker 工具模块
用于文档精排，提升检索准确率
"""

from sentence_transformers import CrossEncoder

# 配置参数
RERANK_MODEL_PATH = "./models/hub/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"  # 本地 reranker 模型路径
# RERANK_MODEL_PATH = "./models/hub/models--jinaai--jina-reranker-v3/snapshots/050e171c4f75dfec5b648ed8470a2475e5a30f30"  # 本地 reranker 模型路径
RERANK_TOP_K = 5  # 默认保留的文档数量


def initialize_reranker(model_path=None):
    """
    初始化 Reranker 模型（从本地加载）
    
    Args:
        model_path: 模型路径，默认使用 RERANK_MODEL_PATH
    
    Returns:
        CrossEncoder 模型实例
    """
    if model_path is None:
        model_path = RERANK_MODEL_PATH
    
    # Jina reranker 需要特殊配置
    if "jina" in model_path.lower():
        print("Initializing Jina reranker with special config...")
        
        reranker = CrossEncoder(
            model_path,
            max_length=512,
            device='cuda',
            local_files_only=True,
            trust_remote_code=True
        )
        
        # 强制设置 pad_token
        if reranker.tokenizer.pad_token is None:
            reranker.tokenizer.pad_token = reranker.tokenizer.eos_token
            print(f"Set pad_token to: {reranker.tokenizer.pad_token}")
        
        # 标记为 Jina 模型（用于后续逐个处理）
        reranker._is_jina = True
    else:
        # BAAI 等其他模型正常初始化
        print("Initializing standard reranker...")
        reranker = CrossEncoder(
            model_path,
            max_length=512,
            device='cuda',
            local_files_only=True
        )
        reranker._is_jina = False
    
    print("Reranker 模型加载完成")
    return reranker


def rerank_documents(query, documents, reranker, top_k=RERANK_TOP_K):
    """
    使用 CrossEncoder 对文档进行精排
    
    Args:
        query: 用户问题
        documents: 初始检索的文档列表
        reranker: CrossEncoder 模型实例
        top_k: 保留的文档数量
    
    Returns:
        精排后的文档列表（附带 rerank_score）
    """
    if not documents:
        return []
    
    if len(documents) <= top_k:
        # 文档数量不足，全部返回
        return documents
    
    # 准备查询-文档对
    query_doc_pairs = [[query, doc.page_content] for doc in documents]
    
    # 计算相关性分数
    # Jina 模型需要逐个处理（避免 padding token 错误）
    if hasattr(reranker, '_is_jina') and reranker._is_jina:
        print(f"Processing {len(query_doc_pairs)} docs one by one (Jina model)...")
        scores = []
        for pair in query_doc_pairs:
            score = reranker.predict([pair])[0]
            scores.append(score)
        import numpy as np
        scores = np.array(scores)
    else:
        # 标准模型批量处理
        scores = reranker.predict(query_doc_pairs)
    
    # 按分数排序
    doc_score_pairs = list(zip(documents, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # 返回 top_k 个文档及其分数
    reranked_docs = []
    for doc, score in doc_score_pairs[:top_k]:
        # 将分数添加到文档元数据
        doc.metadata['rerank_score'] = float(score)
        reranked_docs.append(doc)
    
    return reranked_docs

