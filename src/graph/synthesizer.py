"""
src/graph/synthesizer.py
최종 답변 합성기 노드
Worker A (ERP 처리 결과)와 Worker B (RAG 답변)를 통합하여
비즈니스 이메일 형식의 최종 답변 생성
"""
import logging

from src.graph.state import AgentState

logger = logging.getLogger(__name__)

_STATUS_MESSAGES = {
    "SUCCESS":   "Your request has been processed successfully.",
    "REJECTED":  "Your request has been reviewed and rejected.",
    "BLOCKED_VALIDATION": "Your request could not be processed due to a validation error.",
    "BLOCKED_API":        "Your request could not be processed due to a system error.",
}


def synthesizer_node(state: AgentState) -> dict:
    """
    LangGraph 합성기 노드.
    Worker A/B 결과를 조합해 비즈니스 이메일 초안을 생성합니다.
    """
    intent          = state.get("intent", "QA_ONLY")
    erp_status      = state.get("erp_action_status")
    erp_action      = state.get("erp_action") or {}
    odata_response  = state.get("odata_response") or {}
    rag_answer      = state.get("rag_answer", "")
    errors          = state.get("error_messages", [])

    parts: list[str] = ["Dear Customer,\n"]

    # ── ERP 처리 결과 (Worker A) ─────────────────────────────────────────────
    if intent in ("ACTION_ONLY", "BOTH") and erp_status:
        status_msg = _STATUS_MESSAGES.get(erp_status, f"Status: {erp_status}.")
        parts.append(status_msg)

        if erp_status == "SUCCESS" and odata_response:
            order_id = erp_action.get("order_id", "")
            field    = erp_action.get("field", "")
            new_val  = erp_action.get("new_value", "")
            if order_id:
                parts.append(
                    f"Order {order_id} has been updated"
                    + (f" — {field} set to {new_val}." if field else ".")
                )

        if erp_status == "REJECTED":
            parts.append("No changes have been made to the system.")

    # ── RAG 답변 (Worker B) ──────────────────────────────────────────────────
    if intent in ("QA_ONLY", "BOTH") and rag_answer:
        if intent == "BOTH":
            parts.append("\nRegarding your question:")
        parts.append(rag_answer)

    # ── 오류 메시지 ──────────────────────────────────────────────────────────
    if errors:
        logger.warning("[synthesizer] errors in state: %s", errors)

    # ── 마무리 ───────────────────────────────────────────────────────────────
    parts.append("\nBest regards,\nSAP ERP AI Agent")

    final_response = "\n".join(parts)
    logger.info("[synthesizer] final_response generated (%d chars)", len(final_response))

    return {"final_response": final_response}
