#!/usr/bin/env python3
"""
向现有 FAISS 向量库添加新文档
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
NEW_DOC_DIR = "../document_add"
EXISTING_DB = "faiss_db_jina_add"
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


def add_to_faiss_index(new_documents: List, db_path: str):
    """添加文档到现有向量库"""
    print(f"Adding {len(new_documents)} chunks to existing vector store...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_ID,
        # model_kwargs={"local_files_only": True, "device": "cuda"}
        model_kwargs={"local_files_only": False, "device": "cuda", "trust_remote_code": True}
    )
    
    vectorstore = FAISS.load_local(
        db_path, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    original_count = len(list(vectorstore.docstore._dict.values()))
    vectorstore.add_documents(new_documents)
    new_count = len(list(vectorstore.docstore._dict.values()))
    
    vectorstore.save_local(db_path)
    
    print(f"Added {new_count - original_count} chunks (total: {new_count})")


def main():
    load_dotenv()
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    doc_files = get_document_files(NEW_DOC_DIR)
    new_docs = load_and_chunk_documents(doc_files)
    add_to_faiss_index(new_docs, EXISTING_DB)
    
    print("Done!")


if __name__ == "__main__":
    main()
