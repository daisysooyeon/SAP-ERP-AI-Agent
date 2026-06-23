"""
src/rag/reranker.py
bge-reranker-v2-m3를 사용한 문서 재순위화

- sentence-transformers CrossEncoder 사용 (transformers 5.x 호환)
- CPU/GPU 자동 감지, 싱글턴 캐싱으로 모델 1회만 로딩
- 모델 로드 실패 시 hybrid 검색 순서 그대로 graceful fallback
- top_n은 configs.yaml rag.top_k_rerank에서 읽음
"""

import logging
import re
from langchain_core.documents import Document
from src.config import get_config

logger = logging.getLogger(__name__)

# 메타 일치 계산 시 쿼리 쪽에서 무시할 흔한 불용어 (분모 희석 방지)
_META_STOP = frozenset(
    "the a an of to in on at for and or is are how do does i we you can with "
    "what when where which that this it its my our your be by from as".split()
)

# 싱글턴: bge-reranker 1회만 로딩
_reranker = None          # CrossEncoder 인스턴스 또는 "fallback"
_FALLBACK_SENTINEL = "fallback"
_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

_RRF_K = 60               # 검색 reciprocal-rank 상수 (retriever._rrf_merge 와 동일 관례)


def _minmax(vals: list[float]) -> list[float]:
    """리스트를 [0,1]로 min-max 정규화. 모두 같은 값이면 0.5로 채움."""
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return [0.5] * len(vals)
    return [(v - lo) / (hi - lo) for v in vals]


def _meta_match(query_text: str, doc: Document) -> float:
    """쿼리 키워드와 청크 메타(unit/lesson/section)의 토큰 겹침 비율 [0,1].
    하드 필터가 아니라 소프트 부스트용 신호 — 헤더가 임베딩에 이미 일부 반영하지만,
    여기서 명시적·튜너블하게 보강한다. 불용어는 분모에서 제외해 희석을 줄인다."""
    q = {t for t in re.findall(r"\w+", query_text.lower()) if t not in _META_STOP}
    if not q:
        return 0.0
    meta = " ".join(str(doc.metadata.get(k, "")) for k in ("unit", "lesson", "section"))
    m = set(re.findall(r"\w+", meta.lower()))
    return len(q & m) / len(q)


def _get_reranker():
    global _reranker
    if _reranker is not None:
        return _reranker

    # sentence-transformers의 CrossEncoder로 로딩한다.
    # (구 FlagEmbedding은 transformers 5.x에서 is_torch_fx_available 제거로 import 실패)
    try:
        from sentence_transformers import CrossEncoder
        logger.info("[reranker] Loading %s via CrossEncoder …", _MODEL_NAME)
        _reranker = CrossEncoder(_MODEL_NAME, max_length=512)
        logger.info("[reranker] %s loaded.", _MODEL_NAME)
    except Exception as e:
        logger.warning(
            "[reranker] Failed to load CrossEncoder: %s. "
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
    # CrossEncoder.predict는 쌍 목록에 대한 relevance 점수(높을수록 관련)를 반환한다.
    doc_scores = [float("-inf")] * len(docs)
    for q in all_queries:
        pairs = [(q, doc.page_content) for doc in docs]
        try:
            scores = reranker.predict(pairs)
            if isinstance(scores, float):
                scores = [scores]
            for i, s in enumerate(scores):
                s = float(s)
                if s > doc_scores[i]:
                    doc_scores[i] = s
        except Exception as e:
            logger.warning("[reranker] predict failed for query %r: %s", q[:40], e)

    # ── 점수 블렌딩: cross-encoder 점수 + 검색(RRF) 순위 결합 ────────────────
    # 입력 docs 는 하이브리드 retriever 가 RRF 순으로 넘긴 후보라, 입력 인덱스 i 가 곧
    # 검색 순위다. cross-encoder 가 저평가해도 dense/BM25 가 강하게 민 상위 후보가
    # top_n 밖으로 밀려나지 않도록, 두 신호를 [0,1] 정규화 후 가중합한다.
    #   final = alpha * cross_encoder + (1 - alpha) * 검색_reciprocal_rank
    # alpha=1.0 이면 순수 리랭커(기존 동작). (특히 파편화된 OCR 청크가 dense 상위인데
    #  리랭커가 저평가하는 경우를 구제)
    cfg_rag = get_config().rag
    alpha = cfg_rag.rerank_blend_alpha
    beta  = cfg_rag.meta_boost_beta
    # 블렌딩(alpha<1) 또는 메타 부스트(beta>0)가 켜지면 정규화 결합 점수를 쓴다.
    # (alpha=1.0 & beta>0 이면 final = 정규화 cross_encoder + beta*meta — 순수 리랭커 + 메타)
    if (alpha < 1.0 or beta > 0.0) and len(docs) > 1:
        finite = [s for s in doc_scores if s != float("-inf")]
        floor  = min(finite) if finite else 0.0
        ce     = [s if s != float("-inf") else floor for s in doc_scores]
        rr     = [1.0 / (_RRF_K + i + 1) for i in range(len(docs))]
        ce_n, rr_n = _minmax(ce), _minmax(rr)
        if beta > 0.0:
            meta_q = " ".join(all_queries)   # 모든 쿼리 변형의 토큰을 메타와 대조
            mm_n = _minmax([_meta_match(meta_q, d) for d in docs])
        else:
            mm_n = [0.0] * len(docs)
        final = [alpha * c + (1.0 - alpha) * r + beta * m
                 for c, r, m in zip(ce_n, rr_n, mm_n)]
    else:
        final = doc_scores

    ranked = sorted(zip(final, docs), key=lambda x: x[0], reverse=True)
    result = [doc for _, doc in ranked[:top_n]]

    logger.debug(
        "[reranker] Reranked %d docs → top %d (queries=%d, blend_alpha=%.2f) | scores: %s",
        len(docs), top_n, len(all_queries), alpha,
        [f"{s:.3f}" for s, _ in ranked[:top_n]],
    )
    return result
