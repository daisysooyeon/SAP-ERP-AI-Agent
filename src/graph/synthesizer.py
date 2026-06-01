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
You are an SAP ERP customer service agent composing professional business email replies.

Compose a concise, warm, and professional email reply based on the context below.

Guidelines:
- Start with "Dear Customer,"
- If an ERP action was requested, clearly state what was done (or what could not be done)
- If the ERP action was REJECTED and a "Rejection Reason" is provided, politely convey that
  reason to the customer as the explanation for why the request could not be processed.
  If it was rejected with no reason given, state that the request could not be approved at this time.
- If a knowledge question was answered, include it naturally in the email
- Be factual — only use information provided in the context
- End with "Best regards,\\nSAP ERP Support Team"
- Keep the email under 200 words
"""

_HUMAN = """\
=== CUSTOMER EMAIL ===
{user_input}

=== CONTEXT ===

Intent          : {intent}
ERP Status       : {erp_status}
ERP Action       : {erp_action_summary}
Rejection Reason : {rejection_reason}
RAG Answer       : {rag_answer}
Errors           : {errors}

=== TASK ===
Write the final email reply to the customer based on the context above.
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
) -> str:
    parts = ["Dear Customer,\n"]

    if intent in ("ACTION_ONLY", "BOTH") and erp_status:
        parts.append(_ERP_STATUS_LABELS.get(erp_status, f"ERP status: {erp_status}."))

        if erp_status == "SUCCESS":
            order_id = erp_action.get("order_id", "")
            if order_id:
                parts.append(f"Order {order_id} has been updated successfully.")
        elif erp_status == "REJECTED":
            if rejection_reason:
                parts.append(f"Reason: {rejection_reason}")
            parts.append("No changes have been made to the system.")

    if intent in ("QA_ONLY", "BOTH") and rag_answer:
        if intent == "BOTH":
            parts.append("\nRegarding your question:")
        parts.append(rag_answer)

    parts.append("\nBest regards,\nSAP ERP Support Team")
    return "\n".join(parts)


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
    erp_status = state.get("erp_action_status")
    erp_action = state.get("erp_action") or {}
    rag_answer = state.get("rag_answer") or ""
    errors     = state.get("error_messages", [])
    rejection_reason = state.get("rejection_reason") or ""

    logger.info(
        "[synthesizer] intent=%s erp_status=%s rag_answer_len=%d",
        intent, erp_status, len(rag_answer),
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
        final_response = _template_response(intent, erp_status, erp_action, rag_answer, rejection_reason)
        logger.info("[synthesizer] Fallback template used (%d chars)", len(final_response))

    return {"final_response": final_response}
