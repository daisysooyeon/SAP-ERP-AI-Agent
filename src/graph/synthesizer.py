"""
src/graph/synthesizer.py
Final response synthesizer node

Combines Worker A (ERP processing result) and Worker B (RAG answer) into
a professional business email reply using an LLM.

Falls back to a template-based response if the LLM call fails.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import get_config
from src.graph.state import AgentState

load_dotenv()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM builder (singleton)
# ---------------------------------------------------------------------------

_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is not None:
        return _llm

    cfg     = get_config()
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set.")

    syn_cfg = cfg.models.synthesizer
    _llm = ChatOpenAI(
        model=syn_cfg.name,
        temperature=syn_cfg.temperature,
        openai_api_key=api_key,
        openai_api_base=cfg.openrouter.base_url,
        default_headers={
            "HTTP-Referer": "https://github.com/daisysooyeon/SAP-ERP-AI-Agent",
            "X-Title":      "SAP-ERP-AI-Agent Synthesizer",
        },
    )
    return _llm


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are a customer service agent for the Samsung SDS Logistics Team, composing
professional B2B email replies in English.

Compose a concise, warm, and professional email reply that STRICTLY follows the
template structure below. Do NOT improvise the structure — only the body content
between the fixed opening and closing lines is yours to author.

═════════════ TEMPLATE — REQUIRED STRUCTURE ═════════════

Dear {{Sender Name or "Customer"}},

Hello,

This is the Samsung SDS Logistics Team.

After our internal review of the matter you raised in your email,
{{specific content — 1 to 4 short paragraphs, see body guidance below}}

Please feel free to contact us anytime for any related inquiries.

Thank you.

═════════════ BODY CONTENT GUIDANCE ═════════════

- If an ERP action was requested, clearly state what was done (or what could not
  be done, and why). Reference the order/item numbers concretely.
- If the ERP action was REJECTED and a "Rejection Reason" is provided, politely
  convey that reason to the customer as the explanation. If rejected with no
  reason given, state that the request could not be approved at this time.
- If a knowledge question was answered, include the answer naturally as part of
  the review-result body. Cite the source document briefly when given
  (e.g., "according to the SAP TS460 documentation").
- Be factual — only use information from the provided context. Never invent
  details, dates, or quantities.
- Keep the entire email under 220 words.

═════════════ GREETING NAME ═════════════

If "Sender Name" in the context is provided and not empty, address the customer
by their first name: "Dear {{first name}},". Otherwise use "Dear Customer,".
Use ONLY the first name, even if a full name is provided.
"""

_HUMAN = """\
=== CUSTOMER EMAIL ===
{user_input}

=== CONTEXT ===

Intent           : {intent}
Sender Name      : {sender_name}
ERP Status       : {erp_status}
ERP Action       : {erp_action_summary}
Rejection Reason : {rejection_reason}
RAG Answer       : {rag_answer}
Errors           : {errors}

=== TASK ===
Write the final email reply to the customer based on the context above.
If Sender Name is provided and not empty, address the customer by name
("Dear {{first name}},") instead of the generic "Dear Customer,". Otherwise
use "Dear Customer,".
"""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human",  _HUMAN),
])


# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------

_ERP_STATUS_LABELS = {
    "SUCCESS":                  "ERP update completed successfully.",
    "REJECTED":                 "ERP update was rejected by the approver.",
    "PENDING_APPROVAL":         "ERP update is pending human approval.",
    "FAILED":                   "ERP update failed due to a system error.",
    "BLOCKED_NO_STOCK":         "Request blocked: insufficient stock.",
    "BLOCKED_INVALID_QTY":      "Request blocked: the resulting quantity would be zero or negative.",
    "BLOCKED_SHIPPED":          "Request blocked: item has already shipped.",
    "BLOCKED_EXTRACTION_FAILED":"Request blocked: could not parse the order details.",
    "BLOCKED_VALIDATION":       "Request blocked: validation error.",
    "BLOCKED_NO_DATA":          "Request blocked: order or item not found in the system.",
}


def _erp_action_summary(erp_action: dict, erp_status: str | None) -> str:
    if not erp_action:
        return "None"

    action_type = erp_action.get("action_type", "")
    order_id    = erp_action.get("order_id", "")
    item_no     = erp_action.get("item_no", "")
    status_lbl  = _ERP_STATUS_LABELS.get(erp_status or "", f"Status: {erp_status}")

    base = f"Order {order_id}, Item {item_no} — {action_type} — {status_lbl}"

    if action_type == "CHANGE_QTY" and erp_action.get("new_quantity"):
        base += f" New quantity: {erp_action['new_quantity']}."
    elif action_type == "CHANGE_DATE" and erp_action.get("new_date"):
        base += f" New delivery date: {erp_action['new_date']}."
    elif action_type == "CANCEL_ITEM":
        base += " Item cancellation requested."

    return base


# ---------------------------------------------------------------------------
# Fallback template (LLM call failed)
# ---------------------------------------------------------------------------

def _template_response(
    intent: str,
    erp_status: str | None,
    erp_action: dict,
    rag_answer: str,
    rejection_reason: str = "",
    sender_name: str = "",
) -> str:
    """LLM 호출이 실패해도 Samsung SDS 물류팀 형식의 답장을 결정론적으로 생성한다.
    구조(인사·서명)는 고정, 본문(specific content)만 컨텍스트로 채운다."""
    # Greeting — sender_name이 있으면 first name으로 개인화
    first_name = sender_name.strip().split()[0] if sender_name else ""
    greeting   = f"Dear {first_name}," if first_name else "Dear Customer,"

    body_lines: list[str] = []

    if intent in ("ACTION_ONLY", "BOTH") and erp_status:
        body_lines.append(_ERP_STATUS_LABELS.get(erp_status, f"ERP status: {erp_status}."))

        if erp_status == "SUCCESS":
            order_id = erp_action.get("order_id", "")
            if order_id:
                body_lines.append(f"Order {order_id} has been updated successfully.")
        elif erp_status == "REJECTED":
            if rejection_reason:
                body_lines.append(f"Reason: {rejection_reason}")
            body_lines.append("No changes have been made to the system.")

    if intent in ("QA_ONLY", "BOTH") and rag_answer:
        if intent == "BOTH" and body_lines:
            body_lines.append("")  # 빈 줄로 단락 구분
            body_lines.append("Regarding your question:")
        body_lines.append(rag_answer)

    if not body_lines:
        body_lines.append("we have noted your request and will follow up shortly.")

    body = "\n".join(body_lines)

    return (
        f"{greeting}\n"
        f"\n"
        f"Hello,\n"
        f"\n"
        f"This is the Samsung SDS Logistics Team.\n"
        f"\n"
        f"After our internal review of the matter you raised in your email,\n"
        f"{body}\n"
        f"\n"
        f"Please feel free to contact us anytime for any related inquiries.\n"
        f"\n"
        f"Thank you."
    )


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def synthesizer_node(state: AgentState) -> dict:
    """
    LangGraph synthesizer node.

    Combines Worker A (ERP result) and Worker B (RAG answer) into a
    professional business email using an LLM.
    Falls back to a template if the LLM call fails.
    """
    intent     = state.get("intent", "QA_ONLY")
    user_input = state.get("user_input") or ""
    email_context = state.get("email_context") or {}
    erp_status = state.get("erp_action_status")
    erp_action = state.get("erp_action") or {}
    rag_answer = state.get("rag_answer") or ""
    errors     = state.get("error_messages", [])
    rejection_reason = state.get("rejection_reason") or ""

    # 전처리에서 발신자 이름이 추출됐으면 답장 인사말 개인화에 사용
    sender_name = ""
    if email_context.get("preprocess_ok"):
        sender_name = email_context.get("sender_name", "") or ""

    logger.info(
        "[synthesizer] intent=%s erp_status=%s rag_answer_len=%d sender=%r",
        intent, erp_status, len(rag_answer), sender_name,
    )

    # ── Guard: do NOT compose the final reply while approval is still pending ──
    # For BOTH intent, worker_b routes here directly and can trigger the
    # synthesizer in the same superstep as human_loop's interrupt — producing a
    # premature email before the human approves. Skip until the decision is made;
    # the synthesizer runs again (with SUCCESS/REJECTED) once human_loop resumes.
    if erp_status == "PENDING_APPROVAL":
        logger.info("[synthesizer] Approval pending — skipping reply generation until resumed.")
        return {}

    erp_summary = _erp_action_summary(erp_action, erp_status)

    # ── LLM synthesis ────────────────────────────────────────────────────────
    try:
        llm    = _get_llm()
        chain  = _PROMPT | llm
        result = chain.invoke({
            "user_input":         user_input or "(no email provided)",
            "sender_name":        sender_name or "(unknown)",
            "intent":             intent or "UNKNOWN",
            "erp_status":         erp_status or "N/A",
            "erp_action_summary": erp_summary,
            "rejection_reason":   rejection_reason or "None",
            "rag_answer":         rag_answer or "None",
            "errors":             ", ".join(errors) if errors else "None",
        })
        final_response = result.content.strip()
        logger.info("[synthesizer] LLM response generated (%d chars)", len(final_response))

    except Exception as e:
        logger.warning("[synthesizer] LLM call failed (%s) — using template fallback.", e)
        final_response = _template_response(
            intent, erp_status, erp_action, rag_answer, rejection_reason,
            sender_name=sender_name,
        )
        logger.info("[synthesizer] Fallback template used (%d chars)", len(final_response))

    return {"final_response": final_response}
