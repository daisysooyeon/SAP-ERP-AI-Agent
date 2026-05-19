"""
src/rag/retriever.py
Hybrid Retriever: Dense (ChromaDB/bge-m3) + Sparse (BM25) 앙상블

파라미터는 configs.yaml의 rag 섹션에서 읽습니다:
  rag.dense_weight:    0.6  (ChromaDB 시맨틱 검색 가중치)
  rag.sparse_weight:   0.4  (BM25 키워드 검색 가중치)
  rag.top_k_retrieval: 10   (각 retriever 후보 수)
  paths.chroma_db:     ./chroma_db
  rag.collection_name: sap_manuals

Note: langchain-community 0.4.x에서 EnsembleRetriever가 제거됨 →
      직접 RRF(Reciprocal Rank Fusion) 방식으로 Dense + BM25 결합
"""

import logging
from typing import List

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma

from src.config import get_config

logger = logging.getLogger(__name__)

# 싱글턴: 임베딩 모델과 retriever 컴포넌트는 1회만 로딩
_embedding_model: HuggingFaceBgeEmbeddings | None = None
_vectorstore: Chroma | None = None
_bm25_retriever: BM25Retriever | None = None


def _get_embedding() -> HuggingFaceBgeEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        logger.info("[retriever] Loading BAAI/bge-m3 embedding model …")
        _embedding_model = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-m3",
            encode_kwargs={"batch_size": 32},
        )
        logger.info("[retriever] Embedding model loaded.")
    return _embedding_model


def _get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        cfg = get_config()
        logger.info("[retriever] Connecting to ChromaDB …")
        _vectorstore = Chroma(
            persist_directory=cfg.paths.chroma_db,
            collection_name=cfg.rag.collection_name,
            embedding_function=_get_embedding(),
        )
        logger.info("[retriever] ChromaDB connected.")
    return _vectorstore


def _get_bm25(top_k: int) -> BM25Retriever | None:
    global _bm25_retriever
    if _bm25_retriever is None:
        vs = _get_vectorstore()
        try:
            raw = vs.get()
            docs_for_bm25 = [
                Document(page_content=content, metadata=meta)
                for content, meta in zip(raw["documents"], raw["metadatas"])
            ]
            logger.info("[retriever] BM25 corpus: %d documents", len(docs_for_bm25))
            _bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
            _bm25_retriever.k = top_k
        except Exception as e:
            logger.warning("[retriever] BM25 init failed: %s. Dense-only mode.", e)
            return None
    else:
        _bm25_retriever.k = top_k
    return _bm25_retriever


def _rrf_merge(
    dense_docs: List[Document],
    sparse_docs: List[Document],
    dense_weight: float,
    sparse_weight: float,
    top_k: int,
    k_rrf: int = 60,
) -> List[Document]:
    """
    Reciprocal Rank Fusion (RRF)로 Dense + BM25 결과 병합.
    score(d) = dense_weight / (k_rrf + dense_rank) + sparse_weight / (k_rrf + bm25_rank)
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(dense_docs, 1):
        key = doc.page_content[:100]
        scores[key] = scores.get(key, 0.0) + dense_weight / (k_rrf + rank)
        doc_map[key] = doc

    for rank, doc in enumerate(sparse_docs, 1):
        key = doc.page_content[:100]
        scores[key] = scores.get(key, 0.0) + sparse_weight / (k_rrf + rank)
        doc_map[key] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in ranked[:top_k]]


class HybridRetriever:
    """Dense + BM25 하이브리드 리트리버 (RRF 방식)"""

    def __init__(self):
        cfg = get_config()
        self.top_k         = cfg.rag.top_k_retrieval
        self.dense_weight  = cfg.rag.dense_weight
        self.sparse_weight = cfg.rag.sparse_weight
        self._vs   = _get_vectorstore()
        self._bm25 = _get_bm25(self.top_k)

    def invoke(self, query: str) -> List[Document]:
        # Dense 검색
        try:
            dense_docs = self._vs.similarity_search(query, k=self.top_k)
        except Exception as e:
            logger.warning("[retriever] Dense search failed: %s", e)
            dense_docs = []

        # BM25 검색
        if self._bm25 is not None:
            try:
                sparse_docs = self._bm25.invoke(query)
            except Exception as e:
                logger.warning("[retriever] BM25 search failed: %s", e)
                sparse_docs = []
        else:
            sparse_docs = []

        if not sparse_docs:
            return dense_docs[:self.top_k]

        # RRF 병합
        merged = _rrf_merge(
            dense_docs, sparse_docs,
            self.dense_weight, self.sparse_weight,
            self.top_k,
        )
        logger.debug(
            "[retriever] Dense=%d BM25=%d → merged=%d",
            len(dense_docs), len(sparse_docs), len(merged),
        )
        return merged


# 싱글턴 캐시
_hybrid_retriever: HybridRetriever | None = None


def build_hybrid_retriever(force_rebuild: bool = False) -> HybridRetriever:
    """
    HybridRetriever 싱글턴 반환.
    최초 호출 시 embedding model + ChromaDB + BM25 초기화 (수 초 소요).
    """
    global _hybrid_retriever
    if _hybrid_retriever is None or force_rebuild:
        logger.info("[retriever] Building HybridRetriever …")
        _hybrid_retriever = HybridRetriever()
        logger.info("[retriever] HybridRetriever ready.")
    return _hybrid_retriever
