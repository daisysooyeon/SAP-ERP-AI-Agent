"""
src/rag/reranker.py
bge-reranker-v2-m3를 사용한 문서 재순위화

- CPU 환경에서도 동작 (use_fp16=False)
- 싱글턴 캐싱으로 모델 1회만 로딩
- FlagEmbedding import 실패 시 점수 합산 방식으로 graceful fallback
- top_n은 configs.yaml rag.top_k_rerank에서 읽음
"""

import logging
from langchain_core.documents import Document
from src.config import get_config

logger = logging.getLogger(__name__)

# 싱글턴: bge-reranker 1회만 로딩
_reranker = None          # FlagReranker 인스턴스 또는 "fallback"
_FALLBACK_SENTINEL = "fallback"


def _get_reranker():
    global _reranker
    if _reranker is not None:
        return _reranker

    try:
        from FlagEmbedding import FlagReranker
        logger.info("[reranker] Loading BAAI/bge-reranker-v2-m3 (CPU mode) …")
        _reranker = FlagReranker(
            "BAAI/bge-reranker-v2-m3",
            use_fp16=False,   # CPU 환경 안전 옵션
        )
        logger.info("[reranker] bge-reranker-v2-m3 loaded.")
    except Exception as e:
        logger.warning(
            "[reranker] Failed to load FlagReranker: %s. "
            "Falling back to score-aggregation ordering.",
            e,
        )
        _reranker = _FALLBACK_SENTINEL

    return _reranker


def rerank(
    query: str,
    docs: list[Document],
    top_n: int | None = None,
    queries: list[str] | None = None,
) -> list[Document]:
    """
    bge-reranker-v2-m3로 문서 재순위화 후 top_n 반환.

    queries가 주어지면 Multi-Query Reranking:
      각 문서를 모든 쿼리에 대해 점수 산정 → 문서별 max 점수로 순위 결정.
    queries가 없으면 query 단일 기준으로 순위 결정.

    FlagEmbedding 로드 실패 시 EnsembleRetriever 점수 합산 순서 그대로 반환.

    Args:
        query:   대표 검색 쿼리 (단일 모드 또는 multi-query의 primary)
        docs:    Hybrid Retriever가 반환한 후보 문서 목록
        top_n:   최종 반환 문서 수 (None이면 configs.rag.top_k_rerank 사용)
        queries: 쿼리 변형 목록 (Query Expansion 결과). 있으면 multi-query 모드.
    """
    if top_n is None:
        top_n = get_config().rag.top_k_rerank

    if not docs:
        return []

    reranker = _get_reranker()

    # ── fallback: EnsembleRetriever 점수 합산 순서 유지 ──────────────────────
    if reranker is _FALLBACK_SENTINEL:
        logger.debug("[reranker] Using fallback ordering (top %d of %d)", top_n, len(docs))
        return docs[:top_n]

    # ── 쿼리 목록 결정 ────────────────────────────────────────────────────────
    all_queries = queries if queries else [query]

    # ── bge-reranker-v2-m3: 문서별 max 점수 산정 ─────────────────────────────
    doc_scores = [0.0] * len(docs)
    for q in all_queries:
        pairs = [(q, doc.page_content) for doc in docs]
        try:
            scores = reranker.compute_score(pairs, normalize=True)
            if isinstance(scores, float):
                scores = [scores]
            for i, s in enumerate(scores):
                if s > doc_scores[i]:
                    doc_scores[i] = s
        except Exception as e:
            logger.warning("[reranker] compute_score failed for query %r: %s", q[:40], e)

    ranked = sorted(zip(doc_scores, docs), key=lambda x: x[0], reverse=True)
    result = [doc for _, doc in ranked[:top_n]]

    logger.debug(
        "[reranker] Reranked %d docs → top %d (queries=%d) | scores: %s",
        len(docs), top_n, len(all_queries),
        [f"{s:.3f}" for s, _ in ranked[:top_n]],
    )
    return result
