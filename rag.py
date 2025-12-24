import os
import re
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
import torch
import jieba
from reranker_utils import rerank_documents, initialize_reranker


# # 实验 0: Baseline（基线）
# CONFIG = {
#     "name": "baseline",
#     "top_k": 50,
#     "rerank_top_k": 5,
#     "use_query_expansion": False,
#     "use_hyde": False,
#     "use_doc_filter": False,  # 文档过滤（jieba NER识别+元数据过滤）
#     "use_refine_generation": False,  # Refine生成（分批文档迭代优化答案）
# }

# # 实验 1: Query Expansion（查询扩展）
# CONFIG = {
#     "name": "query_expansion",
#     "top_k": 50,
#     "rerank_top_k": 5,
#     "use_query_expansion": True,
#     "expansion_count": 3,
#     "use_hyde": False,
#     "use_doc_filter": False,
#     "use_refine_generation": False,
# }

# 实验 2: HyDE（假设文档嵌入）
# CONFIG = {
#     "name": "hyde",
#     "top_k": 50,
#     "rerank_top_k": 5,
#     "use_query_expansion": False,
#     "use_hyde": True,
#     "use_doc_filter": False,
#     "use_refine_generation": False,
# }

# 实验 3: Doc Filter（文档过滤：jieba NER识别+元数据过滤）
# CONFIG = {
#     "name": "doc_filter",
#     "top_k": 50,
#     "rerank_top_k": 5,
#     "use_query_expansion": False,
#     "use_hyde": False,
#     "use_doc_filter": True,  # 使用 jieba NER 识别相关文档后过滤
#     "use_refine_generation": False,
# }

# # 实验 4: Query Expansion + Doc Filter（组合策略）
# CONFIG = {
#     "name": "expansion_plus_filter",
#     "top_k": 50,
#     "rerank_top_k": 5,
#     "use_query_expansion": True,
#     "expansion_count": 3,
#     "use_hyde": False,
#     "use_doc_filter": True,
#     "use_refine_generation": False,
# }

# 实验 5: Refine Generation（迭代式生成）
# CONFIG = {
#     "name": "refine_generation",
#     "top_k": 50,
#     "rerank_top_k": 5,  # Refine 需要更多文档
#     "use_query_expansion": False,
#     "use_hyde": False,
#     "use_doc_filter": False,
#     "use_refine_generation": True,  # 分批文档迭代优化答案
# }

# 实验 6: Full Pipeline（所有改进）
# CONFIG = {
#     "name": "full_pipeline",
#     "top_k": 50,
#     "rerank_top_k": 5,
#     "use_query_expansion": True,
#     "expansion_count": 2,
#     "use_hyde": False,
#     "use_doc_filter": True,  # 使用 jieba NER 识别相关文档后过滤
#     "use_refine_generation": True,  # Refine生成（分批文档迭代优化答案）
# }

# ============================================================================
# 配置
# ============================================================================
FAISS_DB_PATH = "faiss_db_qwen"
# EMBED_MODEL_ID = "./models/qwen3-embedding-0.6b"
# FAISS_DB_PATH = "faiss_db_mix"
EMBED_MODEL_ID = "./models/qwen3-embedding-0.6b"
# EMBED_MODEL_ID = "./models/hub/models--jinaai--jina-embeddings-v3/snapshots/f1944de8402dcd5f2b03f822a4bc22a7f2de2eb9"
LOCAL_LLM_PATH = "./models/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
USE_VLLM = False

# ============================================================================
# Prompts
# ============================================================================

PROMPT = PromptTemplate.from_template(
    """你是一名面向个人投资者的财报问答助手。
请严格按照以下指南回答用户问题：
1. 仔细分析问题，识别关键词和核心概念。
2. 从提供的上下文中精确定位相关信息，优先使用完全匹配的内容。
3. 构建回答时，确保包含所有必要的关键词，提高关键词评分。
4. 保持回答与原文的语义相似度，以提高向量相似度评分。
5. 控制回答长度，理想情况下不超过参考上下文长度的1.5倍，最多不超过2.5倍。
6. 对于表格查询或需要多段落/多文档综合的问题，给予特别关注并提供更全面的回答。
7. 如果上下文信息不足，可以进行合理推理，但要明确指出推理部分。
8. 回答应简洁、准确、完整，直接解答问题，避免不必要的解释。
9. 避免输出“检索到的文本块”、“根据”，“信息”等前缀修饰句，直接输出答案即可。
10. 避免使用"根据提供的信息"、"支撑信息显示"等前缀，直接给出答案。
请基于以下检索到的财报内容回答用户问题。
检索内容：
···
{context}
···
用户问题：{input}

回答："""
)

EXPANSION_PROMPT = PromptTemplate.from_template(
    """请从不同角度改写原问题为 {count} 个问题，使用不同关键词和表达方式，以便检索到更多相关文档。只输出问题，不要编号。

原问题：{question}

改写问题："""
)

HYDE_PROMPT = PromptTemplate.from_template(
    """请根据问题生成一段假设的财报文本片段（150-200字），内容应该包含问题的答案。

问题：{question}

财报片段："""
)

REFINE_PROMPT = PromptTemplate.from_template(
    """你是一名面向个人投资者的财报问答助手。
你已经根据之前的财报内容给出了一个初步答案，现在有更多的财报内容可以用来改进答案。

原问题：{question}

已有答案：
{existing_answer}

新增财报内容：
{context}

请基于新增内容改进答案，如果新增内容没有提供更多有价值的信息，则保持原答案不变。

改进后的答案："""
)

QUALITY_CHECK_PROMPT = PromptTemplate.from_template(
    """判断以下答案是否充分回答了用户问题。

用户问题：{question}

当前答案：{answer}

判断标准：
1. 答案是否包含问题要求的关键信息
2. 答案是否具体明确（有数据、有细节）
3. 答案是否完整（没有明显遗漏）

请只输出"充分"或"不充分"。

判断结果："""
)

# ============================================================================
# RAG System
# ============================================================================

class RAGSystem:
    def __init__(self, config: dict = None):
        self.config = config or CONFIG
        
        print(f"\n{'='*60}")
        print(f"Initializing RAG System")
        print(f"Experiment: {self.config['name']}")
        print(f"Config: {self.config}")
        print(f"{'='*60}\n")
        
        load_dotenv()
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_ID,
            # 使用除了jina以外的embeddings模型时请使用下面的配置
            model_kwargs={"local_files_only": True, "device": "cuda"}
            # 使用jina embeddings模型时请使用下面的配置
            # model_kwargs={"local_files_only": False, "device": "cuda", "trust_remote_code": True }
            
        )
        
        self.vectorstore = FAISS.load_local(
            FAISS_DB_PATH, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.config["top_k"]}
        )
        
        self.reranker = initialize_reranker()
        
        self.llm = self._initialize_llm()
        
        if self.config.get("use_doc_filter"):
            self.available_files = self._get_available_files()
            print(f"Available documents: {self.available_files}")
            print("Using NER (jieba) for document filtering\n")
    
    def _initialize_llm(self):
        if USE_VLLM:
            from vllm import LLM, SamplingParams
            
            self.vllm_model = LLM(
                model=LOCAL_LLM_PATH,
                dtype="bfloat16",
                trust_remote_code=True,
                max_model_len=8192,
                gpu_memory_utilization=0.9
            )
            self.vllm_sampling = SamplingParams(
                temperature=0.1,
                max_tokens=1024,
                repetition_penalty=1.3
            )
            return None  # vLLM 不需要返回 LangChain wrapper
        else:
            # 使用 HuggingFace Pipeline
            tokenizer = AutoTokenizer.from_pretrained(
                LOCAL_LLM_PATH,
                trust_remote_code=True,
                local_files_only=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                LOCAL_LLM_PATH,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True
            )
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=1024,
                temperature=0.1, 
                repetition_penalty=1.1,
                return_full_text=False
            )
            return HuggingFacePipeline(pipeline=pipe)
    
    def _generate_text(self, prompt: str) -> str:
        """统一的文本生成接口（支持 vLLM 和 HuggingFace）"""
        if USE_VLLM:
            outputs = self.vllm_model.generate([prompt], self.vllm_sampling)
            return outputs[0].outputs[0].text
        else:
            return self.llm.invoke(prompt)
    
    def _get_available_files(self) -> List[str]:
        docs = list(self.vectorstore.docstore._dict.values())
        files = set()
        for doc in docs:
            if hasattr(doc, 'metadata') and 'dl_meta' in doc.metadata:
                origin = doc.metadata['dl_meta'].get('origin', {})
                if 'filename' in origin:
                    files.add(origin['filename'])
        return sorted(list(files))
    
    def query(self, question: str) -> Dict[str, Any]:
        start_time = time.time()
        
        # Stage 1: Document Filtering (NER Recognition + Metadata Filter)
        filter_files = None
        if self.config.get("use_doc_filter"):
            filter_files = self._recognize_intent_ner(question)
            print(f"Doc filter: {filter_files}")
        
        # Stage 2: Query Rewriting
        queries = [question]
        
        if self.config.get("use_hyde"):
            hyde_doc = self._generate_hyde(question)
            queries.append(hyde_doc)
            print(f"HyDE: {hyde_doc[:60]}...")
        
        if self.config.get("use_query_expansion"):
            expanded = self._expand_query(question)
            queries.extend(expanded)
            print(f"Query expanded to {len(queries)} variants")
        
        # Stage 3: Retrieval
        all_docs_with_scores = []
        seen_contents = set()
        
        for q in queries:
            docs = self._retrieve_with_filter(q, filter_files)  # 每次固定返回 top_k 个
            for doc in docs:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_contents:
                    # 从 metadata 中获取检索分数
                    score = doc.metadata.get('retrieval_score', 0.0)
                    all_docs_with_scores.append((doc, score))
                    seen_contents.add(content_hash)
        
        # 按相似度分数排序（分数越小越相似，FAISS 的 L2 距离）
        all_docs_with_scores.sort(key=lambda x: x[1])
        
        # 截取 top_k * 2 给 reranker（按分数排序后截取）
        max_for_rerank = self.config["top_k"] * 2
        all_docs = [doc for doc, score in all_docs_with_scores[:max_for_rerank]]
        
        print(f"Retrieved {len(all_docs_with_scores)} unique docs, kept top {len(all_docs)} for rerank")
        
        # Stage 4: Rerank
        rerank_top_k = self.config.get("rerank_top_k", 5)
        reranked_docs = rerank_documents(question, all_docs, self.reranker, top_k=rerank_top_k)
        
        # Stage 5: Generation
        if self.config.get("use_refine_generation"):
            response = self._generate_with_refine(question, reranked_docs)
        else:
            context_text = "\n\n".join([d.page_content for d in reranked_docs])
            prompt_text = PROMPT.format(context=context_text, input=question)
            response = self._generate_text(prompt_text)
        
        end_time = time.time()
        
        return {
            "question": question,
            "answer": response,
            "latency_seconds": end_time - start_time,
            "experiment_config": self.config,
            "num_queries": len(queries),
            "num_retrieved": len(all_docs),
            "source_documents": [self._format_doc_info(d) for d in reranked_docs],
            "retrieved_docs": [self._format_doc_info(d) for d in all_docs[:self.config["top_k"]]]
        }
    
    def _recognize_intent_ner(self, question: str) -> Optional[List[str]]:
        """使用 jieba 分词 + 关键词匹配快速识别相关文档"""
        # 1. 使用 jieba 分词
        words = jieba.lcut(question)
        
        # 2. 提取长词（可能是公司名）
        entities = [w for w in words if len(w) >= 2 and not w.isdigit()]
        
        # 3. 匹配文档名
        matched_files = set()
        
        # 直接匹配实体
        for entity in entities:
            for file in self.available_files:
                if entity in file:
                    matched_files.add(file)
        
        # 如果没找到，尝试文档核心名匹配
        if not matched_files:
            for file in self.available_files:
                # 提取文档核心名（去掉年份、后缀等）
                core = re.sub(r'\d{4}年?|\d{4}-\d{4}|年报|财报|财务报告|中期报告|\.pdf|\.docx', '', file).strip()
                if core and len(core) >= 2 and core in question:
                    matched_files.add(file)
        
        return list(matched_files) if matched_files else None
    
    def _generate_hyde(self, question: str) -> str:
        prompt = HYDE_PROMPT.format(question=question)
        return self._generate_text(prompt).strip()
    
    def _expand_query(self, question: str) -> List[str]:
        count = self.config.get("expansion_count", 3)
        prompt = EXPANSION_PROMPT.format(question=question, count=count)
        response = self._generate_text(prompt).strip()
        
        expanded = []
        for line in response.split('\n'):
            line = line.strip()
            if line and line != question:
                expanded.append(line)
        
        return expanded[:count]
    
    def _generate_with_refine(self, question: str, reranked_docs: List[Document]) -> str:
        """使用 Refine 策略生成答案：分批文档迭代优化"""
        if len(reranked_docs) == 0:
            return "检索内容为空，无法回答。"
        
        # 将文档分成 3 组
        n = len(reranked_docs)
        batch_size = max(1, n // 3)
        doc_batches = [
            reranked_docs[i:i+batch_size] 
            for i in range(0, n, batch_size)
        ]
        
        # 确保最多 3 组
        if len(doc_batches) > 3:
            doc_batches = doc_batches[:2] + [reranked_docs[batch_size*2:]]
        
        print(f"Refine generation: {len(doc_batches)} batches ({[len(b) for b in doc_batches]} docs)")
        
        current_answer = None
        
        for i, batch in enumerate(doc_batches):
            batch_context = "\n\n".join([d.page_content for d in batch])
            
            if i == 0:
                # 第一批：直接生成初始答案
                prompt_text = PROMPT.format(context=batch_context, input=question)
                current_answer = self._generate_text(prompt_text)
                print(f"Batch 1/{len(doc_batches)}: Generated initial answer")
            else:
                # 后续批次：基于已有答案改进
                prompt_text = REFINE_PROMPT.format(
                    question=question,
                    existing_answer=current_answer,
                    context=batch_context
                )
                current_answer = self._generate_text(prompt_text)
                print(f"Batch {i+1}/{len(doc_batches)}: Refined answer")
            
            # 最后一批直接输出，中间批次检查质量
            if i < len(doc_batches) - 1:
                is_sufficient = self._check_answer_quality(question, current_answer)
                if is_sufficient:
                    print(f"Answer quality sufficient, stopping at batch {i+1}/{len(doc_batches)}")
                    break
        
        return current_answer
    
    def _check_answer_quality(self, question: str, answer: str) -> bool:
        """使用 LLM 判断答案质量是否充分"""
        prompt = QUALITY_CHECK_PROMPT.format(question=question, answer=answer)
        response = self._generate_text(prompt).strip()
        
        # 判断 LLM 的回答
        return "充分" in response or "足够" in response or "完整" in response
    
    def _retrieve_with_filter(self, query: str, filter_files: Optional[List[str]]) -> List[Document]:
        """先元数据过滤，再向量检索（正确顺序，返回带分数的文档）"""
        
        if filter_files:
            # 定义元数据过滤函数
            def metadata_filter(metadata: dict) -> bool:
                if 'dl_meta' in metadata:
                    origin = metadata.get('dl_meta', {}).get('origin', {})
                    filename = origin.get('filename', '')
                    return any(f in filename for f in filter_files)
                return False
            
            # 先过滤再检索（带分数）
            docs_with_scores = self.vectorstore.similarity_search_with_score(
                query,
                k=self.config["top_k"],
                filter=metadata_filter
            )
            # 将分数保存到文档的 metadata 中
            docs = []
            for doc, score in docs_with_scores:
                doc.metadata['retrieval_score'] = float(score)
                docs.append(doc)
            return docs
        else:
            # 无过滤，全局检索（带分数）
            docs_with_scores = self.vectorstore.similarity_search_with_score(
                query,
                k=self.config["top_k"]
            )
            docs = []
            for doc, score in docs_with_scores:
                doc.metadata['retrieval_score'] = float(score)
                docs.append(doc)
            return docs
    
    def _format_doc_info(self, doc: Document) -> Dict[str, Any]:
        metadata = doc.metadata or {}
        page_info = []
        filename = "unknown"

        if 'dl_meta' in metadata:
            dl_meta = metadata['dl_meta']
            
            if 'origin' in dl_meta and 'filename' in dl_meta['origin']:
                filename = dl_meta['origin']['filename']
            
            if 'doc_items' in dl_meta:
                for item in dl_meta['doc_items']:
                    if 'prov' in item:
                        for prov in item['prov']:
                            if 'page_no' in prov:
                                page_info.append(prov['page_no'])
        
        if filename == "unknown":
            filename = metadata.get('source', 'unknown')
            if '/' in filename:
                filename = filename.split('/')[-1]

        if not page_info:
            p = metadata.get('page', metadata.get('page_number'))
            if p:
                page_info.append(p)
        
        page_info = sorted(list(set(page_info)))

        return {
            "content": doc.page_content,
            "page": page_info,
            "source": filename,
            "score": metadata.get('rerank_score', 0) if 'rerank_score' in metadata else None
        }


# 1. **优先回答**：尽可能根据检索内容回答用户问题
# 2. **部分回答**：如果无法完整回答，请先提供已知的相关信息
# 3. **仔细查找**：认真检查所有检索内容，不要遗漏关键数据
# 4. **明确说明**：如果某个具体数据确实找不到，请说明"检索内容中未提及 [具体数据名称]"
# 5. **禁止编造**：不允许加入外部知识或推测
