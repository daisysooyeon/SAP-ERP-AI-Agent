"""
src/api/schemas.py
FastAPI / LangGraph에서 사용하는 Pydantic 요청/응답 스키마
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal


class ERPActionRequest(BaseModel):
    """Worker A가 LLM을 통해 추출하는 ERP 수정 요청 스키마"""
    # 패딩은 LLM이 아니라 worker_a 후처리 코드가 담당한다. LLM은 원본 숫자를
    # 그대로 추출하면 되므로(예: "6105"), 1~N자리를 허용한다.
    order_id: str = Field(..., description="SAP 영업 오더 번호 (VBELN), 원본 숫자 그대로", pattern=r"^\d{1,10}$")
    item_no: str = Field(..., description="오더 아이템 번호 (POSNR), 원본 숫자 그대로", pattern=r"^\d{1,6}$")
    action_type: Literal["CHANGE_QTY", "CHANGE_DATE", "CANCEL_ITEM", "CHANGE_ADDR", "OTHER"]
    # 절대 수량: "set quantity to 264" → 264
    new_quantity: Optional[int] = Field(None, ge=1, description="변경 후 절대 수량 (양수). 절대 지정일 때만 사용")
    # 상대 수량 변화: "reduce by 50" → -50, "increase by 30" → +30.
    # worker_a가 현재 수량(KWMENG)을 조회해 new_quantity로 환산한다.
    quantity_change: Optional[int] = Field(None, description="상대 수량 변화 (감소=음수, 증가=양수)")
    new_date: Optional[str] = Field(None, description="변경 납기일 (YYYY-MM-DD)")
    new_address: Optional[str] = Field(None, description="변경 배송지 주소 (자유 텍스트)")

    @field_validator("new_quantity")
    @classmethod
    def qty_must_be_positive(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError("수량은 반드시 양수여야 합니다.")
        return v


class RunRequest(BaseModel):
    """POST /api/run 요청 스키마"""
    email_text: str = Field(..., description="처리할 이메일 원문 텍스트")
    thread_id: Optional[str] = Field(None, description="스레드 ID (미지정 시 자동 생성)")


class RunResponse(BaseModel):
    """POST /api/run 응답 스키마"""
    thread_id: str
    intent: Optional[str] = None
    erp_status: Optional[str] = None
    final_response: Optional[str] = None
    requires_approval: bool = False


class ApproveResponse(BaseModel):
    """GET /api/approve 응답 스키마"""
    thread_id: str
    approved: bool
    final_status: str        # SUCCESS | REJECTED | FAILED
    message: str
    errors: list[str] = []
