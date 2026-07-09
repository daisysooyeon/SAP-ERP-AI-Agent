"""
src/graph/business_rules.py
ERP 업무 규칙 엔진

새 규칙 추가 방법:
    RULES 리스트에 BusinessRule 항목을 추가하기만 하면 됩니다.
    worker_a.py는 수정하지 않아도 됩니다.

규칙 평가 순서:
    리스트 순서대로 평가하며, 첫 번째 매칭 규칙의 result를 반환합니다.
    순서가 중요한 규칙(예: SHIPPED → CANCEL_PARTIAL)은 위에 배치하세요.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from src.api.schemas import ERPActionRequest

logger = logging.getLogger(__name__)


@dataclass
class BusinessRule:
    id: str
    description: str
    condition: Callable[[ERPActionRequest, dict], bool]
    result: str


# ---------------------------------------------------------------------------
# 규칙별 condition 함수 (복잡한 로직은 람다 대신 named function으로 분리)
# ---------------------------------------------------------------------------

def _is_fully_shipped(action: ERPActionRequest, v: dict) -> bool:
    return v.get("delivery_status", "") == "C"


def _is_cancel_on_partial(action: ERPActionRequest, v: dict) -> bool:
    return action.action_type == "CANCEL_ITEM" and v.get("delivery_status") == "B"


def _exceeds_available_stock(action: ERPActionRequest, v: dict) -> bool:
    if action.action_type != "CHANGE_QTY" or action.new_quantity is None:
        return False
    available = v.get("available_stock") or 0.0
    current = v.get("quantity")
    # current 미확인 시 보수적으로 new_quantity 전체를 추가분으로 간주
    delta = float(action.new_quantity) if current is None else action.new_quantity - float(current)
    return delta > available


# ---------------------------------------------------------------------------
# 규칙 목록 — 새 규칙은 여기에만 추가
# ---------------------------------------------------------------------------

RULES: list[BusinessRule] = [
    BusinessRule(
        id="SHIPPED_BLOCK",
        description="출하 완료 오더는 수정 불가 (WBSTA=C)",
        condition=_is_fully_shipped,
        result="BLOCKED_SHIPPED",
    ),
    BusinessRule(
        id="CANCEL_PARTIAL",
        description="부분 처리된 아이템은 취소 불가 (WBSTA=B). 수량/날짜 변경은 허용.",
        condition=_is_cancel_on_partial,
        result="BLOCKED_PARTIALLY_PROCESSED",
    ),
    BusinessRule(
        id="NO_STOCK",
        description="가용 재고를 초과하는 수량 증가 불가. 감소는 항상 통과.",
        condition=_exceeds_available_stock,
        result="BLOCKED_NO_STOCK",
    ),
]


# ---------------------------------------------------------------------------
# 규칙 평가 진입점
# ---------------------------------------------------------------------------

def evaluate_rules(
    action: ERPActionRequest,
    validation_result: dict | None,
) -> str | None:
    """
    RULES를 순서대로 평가해 첫 번째 매칭 규칙의 result를 반환합니다.
    모두 통과하면 None을 반환합니다.

    Args:
        action:            Worker A가 추출한 ERP 액션 파라미터
        validation_result: Text-to-SQL로 조회한 DB 스냅샷 (None이면 레코드 없음)
    """
    if validation_result is None:
        logger.warning(
            "[rules] No DB record — BLOCKED_NO_DATA (order=%s item=%s)",
            action.order_id, action.item_no,
        )
        return "BLOCKED_NO_DATA"

    for rule in RULES:
        if rule.condition(action, validation_result):
            logger.warning(
                "[rules] '%s' triggered → %s | %s (order=%s item=%s)",
                rule.id, rule.result, rule.description,
                action.order_id, action.item_no,
            )
            return rule.result

    logger.info("[rules] All rules passed (order=%s item=%s)", action.order_id, action.item_no)
    return None
