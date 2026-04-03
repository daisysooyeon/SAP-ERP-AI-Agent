"""
src/graph/synthesizer.py
최종 답변 합성기 노드
Worker A (ERP 처리 결과)와 Worker B (RAG 답변)를 통합하여
비즈니스 이메일 형식의 최종 답변 생성
"""

from src.graph.state import AgentState


def synthesizer_node(state: AgentState) -> dict:
    """
    LangGraph 합성기 노드 (모델: GPT-4o 또는 Claude 3.5 Sonnet)
    
    입력:
      - user_input: 원본 이메일
      - erp_action_status / odata_response: ERP 처리 결과
      - rag_answer: RAG 기반 정책 답변
    출력:
      - final_response: 인사말 + 처리 결과 + 근거 출처 포함 이메일 초안
    """
    # TODO: Phase 6 구현
    raise NotImplementedError("synthesizer_node: 미구현 (Phase 6)")
