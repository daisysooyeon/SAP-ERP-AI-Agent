"""
src/tools/rag_tools.py
ChromaDB 검색 Tool (Worker B에서 사용)
"""

from src.rag.retriever import build_hybrid_retriever
from src.rag.reranker import rerank


def search_policy_docs(query: str, top_n: int = 3) -> list[dict]:
    """
    Hybrid Retrieval + Reranking으로 정책/매뉴얼 검색
    
    Args:
        query: 검색 쿼리 문자열
        top_n: 최종 반환할 문서 수 (기본 3)
    Returns:
        [{"content": str, "source": str}, ...]
    """
    retriever = build_hybrid_retriever()
    candidates = retriever.invoke(query)
    top_docs = rerank(query, candidates, top_n=top_n)
    return [
        {"content": doc.page_content, "source": doc.metadata.get("source", "unknown")}
        for doc in top_docs
    ]
