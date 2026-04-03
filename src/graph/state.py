"""
src/graph/state.py
AgentState 및 ERPAction TypedDict 정의
"""

from typing import TypedDict, Literal, Optional, List, Any


class ERPAction(TypedDict):
    """ERP 수정 요청 파라미터"""
    order_id: str       # 영업 오더 번호 (VBELN, 10자리)
    item_no: str        # 아이템 번호 (POSNR, 6자리)
    field: str          # 변경 필드 (예: KWMENG, EDATU)
    new_value: Any      # 변경 값
    reason: str         # 변경 사유


class AgentState(TypedDict):
    """LangGraph 그래프 전체가 공유하는 상태 객체"""

    # 입력
    user_input: str

    # 라우터 출력
    intent: Optional[Literal["ACTION_ONLY", "QA_ONLY", "BOTH"]]

    # Worker A
    erp_action: Optional[ERPAction]
    erp_validation_result: Optional[dict]
    erp_action_status: Optional[str]   # PENDING_APPROVAL / BLOCKED_* / SUCCESS / REJECTED
    odata_response: Optional[dict]

    # Worker B
    rag_query: Optional[str]
    retrieved_docs: Optional[List[dict]]
    rag_answer: Optional[str]

    # 최종 출력
    final_response: Optional[str]

    # 메타
    error_messages: List[str]
    requires_human_approval: bool
    human_approved: Optional[bool]
