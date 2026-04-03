"""
src/api/schemas.py
FastAPI / LangGraph에서 사용하는 Pydantic 요청/응답 스키마
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal


class ERPActionRequest(BaseModel):
    """Worker A가 LLM을 통해 추출하는 ERP 수정 요청 스키마"""
    order_id: str = Field(..., description="SAP 영업 오더 번호 (VBELN)", pattern=r"^\d{10}$")
    item_no: str = Field(..., description="오더 아이템 번호 (POSNR)", pattern=r"^\d{6}$")
    action_type: Literal["CHANGE_QTY", "CHANGE_DATE", "CANCEL_ITEM"]
    new_quantity: Optional[int] = Field(None, ge=1, description="변경 수량 (양수)")
    new_date: Optional[str] = Field(None, description="변경 납기일 (YYYY-MM-DD)")

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
