#!/usr/bin/env python3
"""
使用 Docling 构建 FAISS 向量库
DOC_CHUNKS 模式：类 Markdown 文本 + 完整页码元数据
"""

import os
import random
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from langchain_docling.loader import ExportType
from langchain_docling import DoclingLoader
from docling.chunking import HybridChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 配置参数
DOC_DIR = "./scratch"
OUTPUT_DB = "faiss_db_jinamd"
EMBED_MODEL_ID = "./models/hub/models--jinaai--jina-embeddings-v3/snapshots/f1944de8402dcd5f2b03f822a4bc22a7f2de2eb9"
CHUNKER_TOKENIZER = "./models/hub/models--jinaai--jina-embeddings-v3/snapshots/f1944de8402dcd5f2b03f822a4bc22a7f2de2eb9"

SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.pptx', '.html', '.htm', '.md', '.txt', '.asciidoc', '.xml']


def get_document_files(doc_dir: str) -> List[str]:
    """获取目录下所有支持的文档文件"""
    doc_dir = Path(doc_dir)
    doc_files = []
    for ext in SUPPORTED_EXTENSIONS:
        doc_files.extend(doc_dir.glob(f"*{ext}"))
        doc_files.extend(doc_dir.glob(f"*{ext.upper()}"))
    return [str(f) for f in sorted(set(doc_files))]


def load_and_chunk_documents(file_paths: List[str]):
    """加载并分块文档"""
    print(f"Processing {len(file_paths)} files...")
    
    loader = DoclingLoader(
        file_path=file_paths,
        export_type=ExportType.DOC_CHUNKS,
        chunker=HybridChunker(
            tokenizer=CHUNKER_TOKENIZER,
            max_tokens=512
        ),
    )
    
    docs = loader.load()
    print(f"Generated {len(docs)} document chunks")
    
    # 随机预览 3 个文档块
    preview_docs = random.sample(docs, min(3, len(docs)))
    print("\nRandom preview (3 chunks):")
    for i, doc in enumerate(preview_docs, 1):
        print(f"\n--- Chunk {i} ---")
        print(f"Content: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")
    
    return docs


def build_faiss_index(documents: List, output_path: str):
    """构建 FAISS 向量库"""
    print(f"Building FAISS index with {len(documents)} chunks...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_ID,
        # jina
        model_kwargs={"local_files_only": False, "device": "cuda", "trust_remote_code": True}
        # model_kwargs={"local_files_only": True, "device": "cuda"}

    )
    
    vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
    vectorstore.save_local(output_path)
    
    print(f"Vector store saved to: {output_path}/")


def main():
    load_dotenv()
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    doc_files = get_document_files(DOC_DIR)
    docs = load_and_chunk_documents(doc_files)
    build_faiss_index(docs, OUTPUT_DB)
    
    print("Done!")


if __name__ == "__main__":
    main()
