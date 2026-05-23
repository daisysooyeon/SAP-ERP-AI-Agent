"""
src/graph/worker_a.py
Worker A: ERP Transaction Processing Node

Flow:
  ① extract_erp_action()    — LLM structured output → ERPActionRequest (Pydantic)
  ② run_validation_query()  — Text-to-SQL → SQLite execution → erp_validation_result
  ③ check_business_rules()  — Stock / delivery-status checks → BLOCKED_* or pass
  ④ call_sap_odata_patch()  — SAP Sandbox OData v4 PATCH (async → asyncio.run wrapper)
  ⑤ set PENDING_APPROVAL    — Signals human_loop that approval is required
  ⑥ (human_loop handles)   — worker_a is responsible only up to ⑤
"""

from __future__ import annotations

import asyncio
import logging
import os
import re

from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from src.api.schemas import ERPActionRequest
from src.config import get_config
from src.graph.state import AgentState
from src.tools.text_to_sql import run_validation_query
from src.tools.sap_odata_tools import call_sap_odata_patch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ERP Action Extraction — Prompt + LLM
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM = """\
You are an expert at parsing B2B customer emails for SAP ERP order modification requests.

Extract the ERP action parameters from the email below.

Field rules:
- order_id   : 10-digit SAP sales order number (VBELN). Zero-pad if fewer than 10 digits.
               If the email references a number like "4500012345" or "45-0001-2345", normalize to exactly 10 digits.
- item_no    : 6-digit item/line number (POSNR). Zero-pad if needed (e.g. "10" → "000010").
- action_type: One of EXACTLY FIVE values — read the definitions carefully:
               "CHANGE_QTY"  — customer wants to INCREASE or DECREASE the ordered quantity of an item.
               "CHANGE_DATE" — customer wants to RESCHEDULE or UPDATE the delivery/requested date of an item.
               "CANCEL_ITEM" — customer wants to CANCEL or REMOVE an order line item entirely.
               "CHANGE_ADDR" — customer wants to UPDATE or CHANGE the shipping address / ship-to location.
                               Use this whenever the request mentions: shipping address, delivery address,
                               ship-to party, destination address, or similar address-related changes.
               "OTHER"       — any other ERP action that does NOT fit the four above. Use "OTHER" when the
                               request involves any of the following (even if an order/item number is mentioned):
                                 • Unblocking a delivery or removing a delivery block
                                 • Assigning or changing a batch number
                                 • Changing the shipping method, carrier, or route (NOT address)
                                 • Updating payment terms or billing conditions
                                 • Any clarification, inquiry, or system question
                                 • Any action not directly mapping to qty / date / cancel / address
- new_quantity: Integer > 0 (only for CHANGE_QTY). Omit or null for other actions.
- new_date   : ISO date string "YYYY-MM-DD" (only for CHANGE_DATE). Omit or null for other actions.
- new_address : Free-text string with the new shipping address (only for CHANGE_ADDR).
               If the email does not specify a new address, use "[Address to be provided by customer]".
               Omit or null for other actions.

Output ONLY valid JSON matching the required schema. No explanation, no markdown.
"""

_EXTRACTION_HUMAN = "Customer email:\n\n{user_input}"

_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _EXTRACTION_SYSTEM),
    ("human",  _EXTRACTION_HUMAN),
])


def _build_extraction_llm() -> ChatOpenAI:
    """Build the worker_a LLM (OpenRouter) for structured extraction."""
    cfg = get_config()
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set. Check your .env file.")

    wa_cfg = cfg.models.worker_a
    return ChatOpenAI(
        model=wa_cfg.name,
        temperature=wa_cfg.temperature,
        openai_api_key=api_key,
        openai_api_base=cfg.openrouter.base_url,
        default_headers={
            "HTTP-Referer": "https://github.com/daisysooyeon/SAP-ERP-AI-Agent",
            "X-Title": "SAP ERP AI Agent - Worker A",
        },
    )


# Build once at import time
_extraction_llm = _build_extraction_llm()
_extraction_chain = _EXTRACTION_PROMPT | _extraction_llm.with_structured_output(ERPActionRequest)


# ---------------------------------------------------------------------------
# ① Parameter Extraction
# ---------------------------------------------------------------------------

def extract_erp_action(user_input: str) -> ERPActionRequest | None:
    """
    Extract ERPActionRequest from the email using LLM structured output.

    Returns None if extraction fails or validation errors occur.
    """
    logger.info("[worker_a] ① Extracting ERP action parameters …")
    try:
        result: ERPActionRequest = _extraction_chain.invoke({"user_input": user_input})
        # order_id 정규화: 숫자만 남기고 10자리 좌측 제로패딩 (LLM 오패딩 방지)
        raw_id = "".join(c for c in result.order_id if c.isdigit())
        result.order_id = str(int(raw_id)).zfill(10) if raw_id else result.order_id
        logger.info(
            "[worker_a] Extracted: order_id=%s item_no=%s action_type=%s",
            result.order_id, result.item_no, result.action_type,
        )
        return result
    except ValidationError as e:
        logger.error("[worker_a] ERPActionRequest validation failed: %s", e)
        return None
    except Exception as e:
        logger.error("[worker_a] Extraction LLM call failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# ③ Business Rule Checks
# ---------------------------------------------------------------------------

# Delivery status codes that block modification
_BLOCKED_STATUSES = {"C"}  # C = Fully processed / shipped


def check_business_rules(action: ERPActionRequest, validation_result: dict | None) -> str | None:
    """
    Validate ERP business rules against the DB snapshot.

    Returns:
        None                  — all checks passed, proceed
        "BLOCKED_NO_STOCK"    — requested quantity exceeds available stock
        "BLOCKED_SHIPPED"     — item already fully shipped (WBSTA = C)
        "BLOCKED_NO_DATA"     — order/item not found in DB
    """
    logger.info("[worker_a] ③ Checking business rules …")

    if validation_result is None:
        logger.warning("[worker_a] No DB record found for order_id=%s item_no=%s",
                       action.order_id, action.item_no)
        return "BLOCKED_NO_DATA"

    delivery_status = validation_result.get("delivery_status", "")
    available_stock = validation_result.get("available_stock") or 0.0

    # Rule 1: Block if item is already fully shipped
    if delivery_status in _BLOCKED_STATUSES:
        logger.warning(
            "[worker_a] BLOCKED_SHIPPED: delivery_status=%s for order=%s item=%s",
            delivery_status, action.order_id, action.item_no,
        )
        return "BLOCKED_SHIPPED"

    # Rule 2: Block quantity change if requested quantity exceeds available stock
    if action.action_type == "CHANGE_QTY" and action.new_quantity is not None:
        if action.new_quantity > available_stock:
            logger.warning(
                "[worker_a] BLOCKED_NO_STOCK: requested=%d available=%.2f for order=%s item=%s",
                action.new_quantity, available_stock, action.order_id, action.item_no,
            )
            return "BLOCKED_NO_STOCK"

    # Rule 3: CHANGE_ADDR — always passes business rules (address validity is human's responsibility)
    if action.action_type == "CHANGE_ADDR":
        logger.info("[worker_a] CHANGE_ADDR: forwarding to human approval.")
        return None

    logger.info("[worker_a] Business rules passed.")
    return None


# ---------------------------------------------------------------------------
# ④ OData PATCH (async → sync wrapper)
# ---------------------------------------------------------------------------

def _call_odata_sync(action: ERPActionRequest) -> dict:
    """Run the async OData PATCH call synchronously."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Inside an already-running event loop (e.g. Jupyter / FastAPI)
            import nest_asyncio  # type: ignore
            nest_asyncio.apply()
            return loop.run_until_complete(call_sap_odata_patch(action))
        else:
            return loop.run_until_complete(call_sap_odata_patch(action))
    except RuntimeError:
        # No event loop exists yet — create a fresh one
        return asyncio.run(call_sap_odata_patch(action))


# ---------------------------------------------------------------------------
# LangGraph Node
# ---------------------------------------------------------------------------

def worker_a_node(state: AgentState) -> dict:
    """
    LangGraph Worker A node — ERP transaction processing.

    Reads:
        state["user_input"]         — raw email text
        state["error_messages"]     — existing error list (appended on failure)

    Returns a partial AgentState update:
        {
            "erp_action":              ERPActionRequest dict | None,
            "erp_validation_result":   dict | None,
            "erp_action_status":       "PENDING_APPROVAL" | "BLOCKED_*",
            "odata_response":          dict | None,
            "requires_human_approval": bool,
            "error_messages":          [...],
        }
    """
    user_input: str = state["user_input"]
    errors: list[str] = list(state.get("error_messages", []))

    logger.info("[worker_a] ═══════════════ Worker A START ═══════════════")

    # ── ① Extract ERP action parameters ────────────────────────────────────
    action = extract_erp_action(user_input)

    if action is None:
        msg = "worker_a: Failed to extract ERP action parameters from email."
        logger.error("[worker_a] %s", msg)
        errors.append(msg)
        return {
            "erp_action":              None,
            "erp_validation_result":   None,
            "erp_action_status":       "BLOCKED_EXTRACTION_FAILED",
            "odata_response":          None,
            "requires_human_approval": False,
            "error_messages":          errors,
        }

    # ── ② Text-to-SQL validation query ─────────────────────────────────────
    logger.info("[worker_a] ② Running Text-to-SQL validation query …")
    validation_result = run_validation_query(action.order_id, action.item_no)
    logger.info("[worker_a] Validation result: %s", validation_result)

    # ── ③ Business rule checks ──────────────────────────────────────────────
    block_reason = check_business_rules(action, validation_result)

    if block_reason is not None:
        logger.warning("[worker_a] Blocked: %s", block_reason)
        return {
            "erp_action":              action.model_dump(),
            "erp_validation_result":   validation_result,
            "erp_action_status":       block_reason,
            "odata_response":          None,
            "requires_human_approval": False,
            "error_messages":          errors,
        }

    # ── ④ SAP OData PATCH call ─────────────────────────────────────────────
    logger.info("[worker_a] ④ Calling SAP OData PATCH …")
    odata_response: dict | None = None
    try:
        odata_response = _call_odata_sync(action)
        status_code = odata_response.get("status_code", 0)
        logger.info("[worker_a] OData response: status_code=%s", status_code)

        if status_code in (200, 204):
            logger.info("[worker_a] SAP OData PATCH succeeded.")
        elif status_code == 405:
            # SAP Business Accelerator Hub sandbox is read-only (GET only).
            # 405 confirms the endpoint is reachable; write ops require a live S/4HANA system.
            logger.info("[worker_a] SAP sandbox reached (405 expected — write ops not supported in sandbox).")
        else:
            msg = f"worker_a: SAP OData PATCH returned unexpected status {status_code}."
            logger.warning("[worker_a] %s", msg)
            errors.append(msg)

    except Exception as e:
        msg = f"worker_a: OData PATCH call failed: {e}"
        logger.error("[worker_a] %s", msg)
        errors.append(msg)
        # Non-fatal: proceed to approval queue so a human can retry manually.

    # ── ⑤ Queue for human approval ─────────────────────────────────────────
    logger.info("[worker_a] ⑤ Setting PENDING_APPROVAL — routing to human_loop …")
    logger.info("[worker_a] ═══════════════ Worker A END ═══════════════")

    return {
        "erp_action":              action.model_dump(),
        "erp_validation_result":   validation_result,
        "erp_action_status":       "PENDING_APPROVAL",
        "odata_response":          odata_response,
        "requires_human_approval": True,
        "error_messages":          errors,
    }
