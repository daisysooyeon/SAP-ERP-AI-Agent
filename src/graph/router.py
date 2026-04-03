"""
src/graph/router.py
Orchestrator: 이메일 의도 분류 노드 (ACTION_ONLY / QA_ONLY / BOTH)
"""

from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from src.graph.state import AgentState


class RouterOutput(BaseModel):
    """LLM Structured Output 스키마"""
    intent: Literal["ACTION_ONLY", "QA_ONLY", "BOTH"] = Field(
        description="이메일 의도 분류"
    )
    reasoning: str = Field(description="분류 근거 (Chain-of-Thought)")


ROUTER_SYSTEM_PROMPT = """당신은 B2B 영업 이메일을 분석하는 전문가입니다.
이메일을 읽고 아래 세 가지 중 하나로 분류하세요.

- ACTION_ONLY: ERP 데이터 수정/조회 요청만 포함 (예: 수량 변경, 납기 변경)
- QA_ONLY: 사내 규정/정책 질의만 포함 (예: 위약금 조건, 반품 정책)
- BOTH: ERP 수정 요청과 정책 질의가 모두 포함

### 예시 ###
이메일: "PO-2024031200 건의 수량을 500개로 변경해주세요. 긴급 배송 시 추가 비용이 얼마나 되나요?"
분류: BOTH

이메일: "납기를 3월 25일에서 4월 1일로 변경 요청드립니다."
분류: ACTION_ONLY

이메일: "반품 시 위약금 조항이 어떻게 되나요?"
분류: QA_ONLY
"""


def router_node(state: AgentState) -> dict:
    """LangGraph 라우터 노드 — 의도 분류 후 intent 반환"""
    # TODO: get_llm("router") 구현 후 연결 (Qwen2.5-3B-Instruct)
    raise NotImplementedError("router_node: LLM 연결 미구현")
