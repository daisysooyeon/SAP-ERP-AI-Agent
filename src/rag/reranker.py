"""
src/rag/reranker.py
bge-reranker-v2-m3를 사용한 문서 재순위화 (top-3 선별)
"""

from FlagEmbedding import FlagReranker
from langchain.schema import Document

_reranker: FlagReranker | None = None


def _get_reranker() -> FlagReranker:
    global _reranker
    if _reranker is None:
        _reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
    return _reranker


def rerank(query: str, docs: list[Document], top_n: int = 3) -> list[Document]:
    """
    bge-reranker-v2-m3로 문서 재순위화 후 top_n 반환
    
    Args:
        query: 검색 쿼리
        docs: Hybrid Retriever가 반환한 후보 문서 목록 (top-10)
        top_n: 최종 반환 문서 수 (기본 3)
    """
    if not docs:
        return []

    reranker = _get_reranker()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.compute_score(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]
