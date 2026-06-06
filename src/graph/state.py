"""
src/graph/state.py
AgentState 및 ERPAction TypedDict 정의
"""

import operator
from typing import Annotated, TypedDict, Literal, Optional, List, Any


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

    # 전처리 결과 (LangGraph 진입 전 entry point에서 채워짐)
    # 자연어 이메일을 구조화한 EmailContext의 .model_dump() 결과 — 다음 필드 포함:
    #   sender_name / sender_email / sender_company / recipient / subject / language
    #   cleaned_body / request_summary
    #   mentions_action / mentions_question
    #   order_ids / item_nos
    #   preprocess_ok / error
    # 없으면(None) 다운스트림 노드는 원본 user_input을 직접 파싱하는 기존 동작으로 폴백.
    email_context: Optional[dict]

    # 라우터 출력
    intent: Optional[Literal["ACTION_ONLY", "QA_ONLY", "BOTH"]]

    # Worker A
    erp_action: Optional[ERPAction]
    erp_validation_result: Optional[dict]
    erp_action_status: Optional[str]   # PENDING_APPROVAL / BLOCKED_* / SUCCESS / REJECTED
    odata_response: Optional[dict]
    rejection_reason: Optional[str]    # 승인자가 Slack에서 입력한 거절 사유 (REJECTED 시)

    # Worker B
    rag_query: Optional[str]
    rag_queries: Optional[List[str]]
    retrieved_docs: Optional[List[dict]]
    rag_answer: Optional[str]

    # 최종 출력
    final_response: Optional[str]

    # 메타
    error_messages: Annotated[List[str], operator.add]
    requires_human_approval: bool
    human_approved: Optional[bool]
