"""
src/graph/worker_a.py
Worker A: ERP 트랜잭션 처리 노드
흐름: 파라미터 추출 → 검증(Text-to-SQL) → OData Call → Human Approval 큐잉 → DB Update
"""

from src.graph.state import AgentState


def worker_a_node(state: AgentState) -> dict:
    """
    LangGraph Worker A 노드
    
    처리 흐름:
      ① Parameter Extraction  — LLM으로 ERPActionRequest 추출 + Pydantic 검증
      ② Text-to-SQL           — dd03l 스키마 기반 검증 쿼리 생성 → SQLite 실행
      ③ Usage Check           — 재고(MARD) / 출하상태(VBUP) / 납기일(VBEP) 검증
      ④ OData Call            — SAP CE_SALESORDER_0001 API PATCH 호출
      ⑤ Human Approval        — interrupt 전 PENDING_APPROVAL 상태 설정
      ⑥ DB Update             — human_loop에서 승인 후 SQLite UPDATE
    """
    # TODO: Phase 3 구현
    raise NotImplementedError("worker_a_node: 미구현 (Phase 3)")
