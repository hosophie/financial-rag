#!/usr/bin/env python
"""Quick Chroma smoke test for company metadata filtering.

This script intentionally avoids downloading embedding models. It uses a tiny
deterministic keyword/hash embedding so the test focuses on Chroma collection
creation, vector lookup, metadata filtering, and document-content filtering.
"""

from __future__ import annotations

import hashlib
import math
import shutil
from pathlib import Path
from typing import Any

import chromadb


ROOT_DIR = Path(__file__).resolve().parent
DB_DIR = ROOT_DIR / "chroma_db"
COLLECTION_NAME = "financial_report_v2_smoke"
EMBED_DIM = 96

COMPANIES = ["宁德时代", "比亚迪", "贵州茅台", "腾讯控股", "隆基绿能"]

FEATURE_GROUPS = {
    "研发": ["研发", "技术", "专利", "创新", "实验室", "算法", "平台"],
    "新能源": ["新能源", "动力电池", "储能", "光伏", "组件", "硅片", "电动车", "碳酸锂"],
    "现金流": ["现金流", "经营活动", "现金", "回款", "应收", "资本开支", "现金等价物"],
    "营收": ["营收", "收入", "销售", "订单", "增长", "同比", "市场份额"],
    "毛利率": ["毛利率", "毛利", "成本", "原材料", "价格", "产品结构"],
    "风险": ["风险", "竞争", "政策", "汇率", "供应链", "减值", "价格波动"],
    "国际化": ["海外", "国际", "出口", "全球", "欧洲", "东南亚"],
    "渠道": ["渠道", "经销商", "客户", "门店", "直销", "会员"],
}

DOCS = [
    {
        "company": "宁德时代",
        "year": "2024",
        "topic": "研发",
        "content": "宁德时代2024年持续加大研发投入，重点推进麒麟电池、钠离子电池和储能系统技术。公司新增多项动力电池专利，研发团队围绕安全性、能量密度和快充体验开展验证。",
    },
    {
        "company": "宁德时代",
        "year": "2024",
        "topic": "现金流",
        "content": "宁德时代经营活动现金流保持净流入，客户回款节奏改善，应收账款周转效率提升。公司控制资本开支节奏，同时保障储能和新能源电池产线扩产资金。",
    },
    {
        "company": "宁德时代",
        "year": "2024",
        "topic": "毛利率",
        "content": "宁德时代毛利率受碳酸锂价格回落和产品结构优化影响有所修复。动力电池规模效应增强，海外客户占比提升也改善了单位制造成本。",
    },
    {
        "company": "宁德时代",
        "year": "2024",
        "topic": "风险",
        "content": "宁德时代提示新能源补贴政策变化、原材料价格波动和海外贸易壁垒可能影响盈利。公司通过长期采购协议和全球化产能布局降低供应链风险。",
    },
    {
        "company": "比亚迪",
        "year": "2024",
        "topic": "研发",
        "content": "比亚迪研发投入集中在刀片电池、智能驾驶平台和混动系统升级。公司加强新能源汽车核心零部件自研能力，以提升整车续航、安全和成本控制。",
    },
    {
        "company": "比亚迪",
        "year": "2024",
        "topic": "营收",
        "content": "比亚迪新能源汽车销量增长带动营收提升，乘用车出口和高端品牌车型贡献增量。公司在国内市场保持领先，同时加快欧洲和东南亚销售网络建设。",
    },
    {
        "company": "比亚迪",
        "year": "2024",
        "topic": "现金流",
        "content": "比亚迪经营活动现金流受整车销量增长和供应链议价能力支撑。公司存货管理改善，电池和汽车业务回款稳定，现金等价物余额保持充足。",
    },
    {
        "company": "比亚迪",
        "year": "2024",
        "topic": "风险",
        "content": "比亚迪面临新能源汽车价格竞争、海外认证周期和汇率波动风险。管理层强调通过垂直整合和规模化制造对冲行业降价压力。",
    },
    {
        "company": "贵州茅台",
        "year": "2024",
        "topic": "营收",
        "content": "贵州茅台2024年酒类销售收入稳健增长，核心单品茅台酒保持供需紧平衡。直营渠道和数字化平台提升了终端触达能力。",
    },
    {
        "company": "贵州茅台",
        "year": "2024",
        "topic": "毛利率",
        "content": "贵州茅台毛利率维持高位，产品结构以高端白酒为主，吨酒价格稳定。公司通过精细化生产和渠道管理保持盈利质量。",
    },
    {
        "company": "贵州茅台",
        "year": "2024",
        "topic": "现金流",
        "content": "贵州茅台经营活动现金流充沛，预收款和合同负债体现经销商打款积极。公司分红能力较强，现金储备覆盖生产扩建和渠道投入。",
    },
    {
        "company": "贵州茅台",
        "year": "2024",
        "topic": "风险",
        "content": "贵州茅台关注白酒消费场景变化、渠道库存波动和价格预期管理风险。公司通过稳定投放节奏和强化品牌建设维护长期需求。",
    },
    {
        "company": "腾讯控股",
        "year": "2024",
        "topic": "研发",
        "content": "腾讯控股持续投入AI大模型、云计算和安全技术研发。公司将算法能力用于广告推荐、企业服务和游戏内容生产，提升平台效率。",
    },
    {
        "company": "腾讯控股",
        "year": "2024",
        "topic": "营收",
        "content": "腾讯控股收入来自增值服务、网络广告、金融科技和企业服务。视频号广告加载率提升，小游戏生态和云业务恢复增长。",
    },
    {
        "company": "腾讯控股",
        "year": "2024",
        "topic": "现金流",
        "content": "腾讯控股自由现金流保持稳健，经营活动现金流覆盖服务器投入、内容采购和股东回报。公司控制低效项目支出，提高资本配置效率。",
    },
    {
        "company": "腾讯控股",
        "year": "2024",
        "topic": "风险",
        "content": "腾讯控股披露监管政策、游戏版号、数据安全和云服务竞争风险。公司通过合规审查和技术安全体系降低平台运营不确定性。",
    },
    {
        "company": "隆基绿能",
        "year": "2024",
        "topic": "研发",
        "content": "隆基绿能研发聚焦高效电池技术、硅片工艺和光伏组件可靠性。公司推进HPBC和其他新型电池路线，以提升转换效率和产品竞争力。",
    },
    {
        "company": "隆基绿能",
        "year": "2024",
        "topic": "营收",
        "content": "隆基绿能组件出货规模扩大，但光伏产业链价格下行压制营收弹性。海外市场需求增长，分布式项目和大型电站客户贡献订单。",
    },
    {
        "company": "隆基绿能",
        "year": "2024",
        "topic": "现金流",
        "content": "隆基绿能经营活动现金流承压，主要因为组件价格下降、客户账期拉长和库存去化。公司减少非必要资本开支，提升回款管理力度。",
    },
    {
        "company": "隆基绿能",
        "year": "2024",
        "topic": "风险",
        "content": "隆基绿能面临光伏行业产能过剩、硅料价格波动、海外贸易政策和技术路线迭代风险。公司通过研发和全球渠道分散经营压力。",
    },
]


def stable_index(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % EMBED_DIM


def add_feature(vector: list[float], key: str, weight: float) -> None:
    vector[stable_index(key)] += weight


def iter_char_ngrams(text: str, n: int = 2) -> list[str]:
    compact = "".join(ch for ch in text if not ch.isspace())
    return [compact[i : i + n] for i in range(max(0, len(compact) - n + 1))]


def embed_text(text: str) -> list[float]:
    vector = [0.0] * EMBED_DIM

    for company in COMPANIES:
        if company in text:
            add_feature(vector, f"company:{company}", 1.2)

    for group, keywords in FEATURE_GROUPS.items():
        matched = False
        for keyword in keywords:
            if keyword in text:
                add_feature(vector, f"kw:{keyword}", 1.0)
                matched = True
        if matched:
            add_feature(vector, f"topic:{group}", 2.0)

    for gram in iter_char_ngrams(text):
        add_feature(vector, f"gram:{gram}", 0.03)

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def build_records() -> tuple[list[str], list[str], list[dict[str, Any]], list[list[float]]]:
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    embeddings: list[list[float]] = []

    for index, item in enumerate(DOCS, start=1):
        doc_id = f"fake-report-{index:02d}"
        content = item["content"]
        metadata = {
            "doc_id": doc_id,
            "company": item["company"],
            "year": item["year"],
            "topic": item["topic"],
        }
        ids.append(doc_id)
        documents.append(content)
        metadatas.append(metadata)
        embeddings.append(embed_text(content))

    return ids, documents, metadatas, embeddings


def reset_collection():
    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)

    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    ids, documents, metadatas, embeddings = build_records()
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    return collection


def query_collection(
    collection,
    text: str,
    n_results: int = 5,
    where: dict[str, Any] | None = None,
    where_document: dict[str, Any] | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "query_embeddings": [embed_text(text)],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where is not None:
        kwargs["where"] = where
    if where_document is not None:
        kwargs["where_document"] = where_document
    return collection.query(**kwargs)


def get_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    for doc_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
        rows.append(
            {
                "id": doc_id,
                "document": document,
                "metadata": metadata,
                "distance": distance,
            }
        )
    return rows


def print_results(title: str, query: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = get_rows(result)
    print(f"\n{'=' * 80}")
    print(title)
    print(f"Query: {query}")
    print(f"Result count: {len(rows)}")
    print("-" * 80)

    if not rows:
        print("No results")
        return rows

    for rank, row in enumerate(rows, start=1):
        metadata = row["metadata"]
        content = row["document"]
        print(
            f"#{rank} distance={row['distance']:.4f} "
            f"company={metadata['company']} topic={metadata['topic']} "
            f"doc_id={metadata['doc_id']}"
        )
        print(f"   {content[:110]}")

    return rows


def print_conclusion(
    global_rows: list[dict[str, Any]],
    company_rows: list[dict[str, Any]],
    company_cash_rows: list[dict[str, Any]],
) -> None:
    global_companies = sorted({row["metadata"]["company"] for row in global_rows})
    global_topics = [row["metadata"]["topic"] for row in global_rows[:3]]
    company_filter_ok = bool(company_rows) and all(
        row["metadata"]["company"] == "宁德时代" for row in company_rows
    )
    company_cash_ok = bool(company_cash_rows) and all(
        row["metadata"]["company"] == "宁德时代" and "现金流" in row["document"]
        for row in company_cash_rows
    )
    false_positive = [
        row["metadata"]["company"]
        for row in company_rows
        if row["metadata"]["company"] != "宁德时代"
    ]

    print(f"\n{'=' * 80}")
    print("Quick conclusion")
    print(f"- Collection path: {DB_DIR}")
    print(f"- Inserted records: {len(DOCS)}")
    print(f"- Global query returned companies: {', '.join(global_companies)}")
    print(f"- Global top-3 topics: {', '.join(global_topics)}")
    print(f"- Company metadata filter only returned 宁德时代: {company_filter_ok}")
    print(f"- Company + document keyword filter matched 宁德时代现金流 docs: {company_cash_ok}")
    if false_positive:
        print(f"- Obvious false positives under company filter: {false_positive}")
    else:
        print("- Obvious false positives under company filter: none")
    print(
        "- Note: embedding is a deterministic smoke-test embedding, "
        "not a production semantic model."
    )


def main() -> None:
    collection = reset_collection()
    print(f"Created collection: {COLLECTION_NAME}")
    print(f"Collection count: {collection.count()}")

    global_query = "研发投入和新能源业务进展"
    company_query = "宁德时代研发投入和新能源电池技术"
    cash_query = "宁德时代现金流和回款情况"

    global_rows = print_results(
        "1) Global vector search without filters",
        global_query,
        query_collection(collection, global_query, n_results=6),
    )
    company_rows = print_results(
        '2) Vector search with metadata filter where={"company": "宁德时代"}',
        company_query,
        query_collection(
            collection,
            company_query,
            n_results=4,
            where={"company": "宁德时代"},
        ),
    )
    company_cash_rows = print_results(
        '3) Metadata filter + document filter where_document={"$contains": "现金流"}',
        cash_query,
        query_collection(
            collection,
            cash_query,
            n_results=4,
            where={"company": "宁德时代"},
            where_document={"$contains": "现金流"},
        ),
    )

    print_conclusion(global_rows, company_rows, company_cash_rows)


if __name__ == "__main__":
    main()
