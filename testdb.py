# #!/usr/bin/env python3
"""
查看向量库中每个文件的分块数量统计
"""

import os
from collections import defaultdict
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# 配置参数
FAISS_DB_PATH = "faiss_db_qwen"
# EMBED_MODEL_ID = "./models/hub/models--jinaai--jina-embeddings-v3/snapshots/f1944de8402dcd5f2b03f822a4bc22a7f2de2eb9"
EMBED_MODEL_ID = "./models/qwen3-embedding-0.6b"


def extract_filename(metadata):
    """从元数据中提取文件名"""
    if 'source' in metadata:
        source = metadata['source']
        return source.split('/')[-1] if '/' in source else source
    
    if 'dl_meta' in metadata and 'origin' in metadata['dl_meta']:
        origin = metadata['dl_meta']['origin']
        if 'filename' in origin:
            return origin['filename']
    
    return 'unknown'


def main():
    """统计向量库中每个文件的分块数量"""
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    print("="*80)
    print("📚 加载向量库...")
    print("="*80)
    print(f"Vector DB: {FAISS_DB_PATH}")
    print(f"Embedding: {EMBED_MODEL_ID.split('/')[-1][:50]}...\n")
    
    # 初始化 Embedding 模型
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_ID,
        model_kwargs={"local_files_only": True, "device": "cuda"}
        # model_kwargs={"local_files_only": False, "device": "cuda", "trust_remote_code": True}
    )
    
    # 加载向量库
    vectorstore = FAISS.load_local(
        FAISS_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    # 获取所有文档
    all_docs = list(vectorstore.docstore._dict.values())
    
    print(f"✅ 加载完成！总文档块数: {len(all_docs)}\n")
    
    # 统计每个文件的分块数量
    file_chunks = defaultdict(int)
    file_char_count = defaultdict(int)
    
    for doc in all_docs:
        filename = extract_filename(doc.metadata)
        file_chunks[filename] += 1
        file_char_count[filename] += len(doc.page_content)
    
    # 按分块数量排序（降序）
    sorted_files = sorted(file_chunks.items(), key=lambda x: x[1], reverse=True)
    
    print("="*80)
    print("📊 每个文件的分块数量统计")
    print("="*80)
    print(f"{'文件名':<50} {'分块数':>8} {'平均长度':>10}")
    print("-"*80)
    
    for filename, count in sorted_files:
        avg_len = file_char_count[filename] // count if count > 0 else 0
        print(f"{filename:<50} {count:>8} {avg_len:>10}")
    
    print("-"*80)
    print(f"{'总计':<50} {sum(file_chunks.values()):>8} {sum(file_char_count.values()) // len(all_docs):>10}")
    
    print("\n" + "="*80)
    print("✅ 统计完成！")
    print("="*80)


if __name__ == "__main__":
    main()

