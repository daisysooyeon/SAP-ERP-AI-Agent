"""
src/graph/human_loop.py
Human-in-the-Loop 노드
- LangGraph interrupt_before=["human_loop"]로 설정
- 담당자 승인 후 SQLite UPDATE 실행
"""

from src.graph.state import AgentState


def human_loop_node(state: AgentState) -> dict:
    """
    LangGraph Human-in-the-Loop 노드
    
    이 노드에 도달 = FastAPI /api/approve 에서 그래프를 재개한 상태
    (interrupt로 일시 정지 → 슬랙 버튼 클릭 → 그래프 재개)
    
    - human_approved == True  → SQLite UPDATE → SUCCESS
    - human_approved == False → REJECTED
    """
    if not state.get("human_approved"):
        return {"erp_action_status": "REJECTED"}

    # TODO: update_sqlite(state["erp_action"]) 구현
    raise NotImplementedError("human_loop_node: SQLite UPDATE 미구현 (Phase 5)")
