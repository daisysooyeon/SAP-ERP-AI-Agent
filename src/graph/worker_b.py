"""
src/graph/worker_b.py
Worker B: RAG 검색 노드
흐름: 키워드 추출 → Hybrid Retrieval → Reranking → 답변 생성
"""

from src.graph.state import AgentState


def worker_b_node(state: AgentState) -> dict:
    """
    LangGraph Worker B 노드
    
    처리 흐름:
      ① 이메일에서 검색 쿼리 추출 (LLM)
      ② Hybrid Retriever (Dense 0.6 + BM25 0.4) — top-10 후보 수집
      ③ bge-reranker-v2-m3 — top-3 재순위화
      ④ top-3 컨텍스트 기반 답변 생성 (LLM)
    """
    # TODO: Phase 4 구현
    raise NotImplementedError("worker_b_node: 미구현 (Phase 4)")
