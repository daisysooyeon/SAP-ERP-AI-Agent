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

from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from src.api.schemas import ERPActionRequest
from src.config import get_config
from src.graph.state import AgentState
from src.tools.text2sql import run_validation_query
from src.tools.sap_odata_tools import call_sap_odata_patch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ERP Action Extraction — Prompt + LLM
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM = """\
You are an expert at parsing B2B customer emails for SAP ERP order modification requests.

Extract the ERP action parameters from the email below.

CRITICAL — reading numbers correctly:
  A single email often contains SEVERAL numbers: the order number, the item/line
  number, and a quantity. Map each to the correct field and copy EVERY digit
  exactly as written — do NOT add, drop, reorder, pad, or invent any digit.
  Do NOT zero-pad — the system pads automatically. Just return the raw digits.
  Ignore unrelated how-to / configuration / inquiry sentences; they never change
  the order or item number.

Field rules:
- order_id   : SAP sales order number (VBELN) — introduced by "order", "order #",
               "sales order", or "PO". Copy its digits EXACTLY as written.
               Strip separators like "-" only. It is NOT the item number and NOT a
               quantity. No padding.
- item_no    : Item / line number (POSNR) — introduced by "item", "line", or "position".
               Copy its digits exactly as written. No padding.
               Never confuse it with the order number or a quantity.
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
- Quantity (only for CHANGE_QTY) — distinguish ABSOLUTE vs RELATIVE, fill exactly ONE:
    • new_quantity    : the ABSOLUTE target quantity, when the customer states the final
                        value — e.g. "set quantity to 264", "change quantity to 100",
                        "make it 80 units". Integer > 0.
    • quantity_change : the RELATIVE change, when the customer states an adjustment rather
                        than a final value — e.g. "reduce by 50" → -50, "decrease by 20" → -20,
                        "increase by 30" → +30, "add 10 more units" → +10.
                        Use a NEGATIVE number for reductions, POSITIVE for increases.
    Never put a relative adjustment into new_quantity. "reduce by 50" is quantity_change=-50,
    NOT new_quantity=50. Leave the other field null.
- new_date   : ISO date string "YYYY-MM-DD" (only for CHANGE_DATE). Omit or null for other actions.
- new_address : Free-text string with the new shipping address (only for CHANGE_ADDR).
               If the email does not specify a new address, use "[Address to be provided by customer]".
               Omit or null for other actions.

Output ONLY valid JSON matching the required schema. No explanation, no markdown.
"""

_EXTRACTION_HUMAN = """\
Customer email:

{user_input}{preprocess_hint}"""

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

def _build_preprocess_hint(email_context: dict | None) -> str:
    """전처리 결과에서 worker_a 추출에 도움이 되는 hint를 만든다.
    LLM은 여전히 자기 판단으로 최종값을 정하지만(=order_ids 중 어느 게 진짜 order_id인지,
    item_nos 중 어느 게 진짜 item_no인지는 문맥에 의존), 후보 풀을 미리 좁혀주면 다른
    숫자(수량, 날짜의 일부)와 헷갈리는 실수를 줄일 수 있다."""
    if not email_context or not email_context.get("preprocess_ok"):
        return ""

    parts = ["\n\n---\nPreprocessor hints (use as a cross-check, not authoritative):"]
    if email_context.get("request_summary"):
        parts.append(f"- Request summary: {email_context['request_summary']}")
    if email_context.get("order_ids"):
        parts.append(f"- Candidate order IDs detected in email: {email_context['order_ids']}")
    if email_context.get("item_nos"):
        parts.append(f"- Candidate item numbers detected in email: {email_context['item_nos']}")
    return "\n".join(parts)


def extract_erp_action(user_input: str, email_context: dict | None = None) -> ERPActionRequest | None:
    """
    Extract ERPActionRequest from the email using LLM structured output.

    Args:
        user_input:    raw email text. Used as the extraction input only when the
                       preprocessor produced no usable cleaned_body.
        email_context: optional preprocessor result. When preprocess_ok and a
                       cleaned_body is present, the cleaned_body (greetings/signatures
                       stripped, but order/item/quantity digits preserved) becomes the
                       extraction input; its summary/entity fields are still passed as
                       soft cross-check hints. Mirrors worker_b's pattern.

    Returns None if extraction fails or validation errors occur.
    """
    # 전처리가 만든 cleaned_body가 있으면 추출 입력으로 사용한다(worker_b와 동일 패턴).
    # 인사·서명 노이즈가 제거돼 더 안정적이며, cleaned_body는 주문/아이템/수량 숫자를
    # 원문 그대로 보존한다. 없거나 전처리 실패 시 원본 user_input으로 폴백.
    extraction_input = user_input
    used_cleaned = bool(
        email_context
        and email_context.get("preprocess_ok")
        and email_context.get("cleaned_body")
    )
    if used_cleaned:
        extraction_input = email_context["cleaned_body"]

    logger.info("[worker_a] ① Extracting ERP action parameters … (input=%s%s)",
                "cleaned_body" if used_cleaned else "raw",
                ", with hints" if email_context else "")
    try:
        result: ERPActionRequest = _extraction_chain.invoke({
            "user_input": extraction_input,
            "preprocess_hint": _build_preprocess_hint(email_context),
        })
        # 패딩은 여기(결정론적 코드)서 담당한다. LLM은 원본 숫자만 추출하므로
        # 끝자리를 흘리는 오패딩이 발생하지 않는다.
        # order_id → 10자리, item_no → 6자리 좌측 제로패딩.
        raw_id = "".join(c for c in result.order_id if c.isdigit())
        result.order_id = str(int(raw_id)).zfill(10) if raw_id else result.order_id
        raw_item = "".join(c for c in result.item_no if c.isdigit())
        result.item_no = str(int(raw_item)).zfill(6) if raw_item else result.item_no
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
        None                          — all checks passed, proceed
        "BLOCKED_NO_STOCK"            — additional quantity (delta) exceeds available stock
        "BLOCKED_SHIPPED"            — item already fully shipped (WBSTA = C)
        "BLOCKED_PARTIALLY_PROCESSED" — cancellation of an item whose goods movement
                                        already started (WBSTA = B)
        "BLOCKED_NO_DATA"            — order/item not found in DB
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

    # Rule 1b: An item cannot be CANCELLED once goods movement has started.
    # Full shipment (WBSTA=C) is already caught by Rule 1 (BLOCKED_SHIPPED); here we
    # additionally block partial processing (WBSTA=B) — but only for cancellations.
    # Other actions (qty/date) on a partially processed item remain allowed.
    if action.action_type == "CANCEL_ITEM" and delivery_status == "B":
        logger.warning(
            "[worker_a] BLOCKED_PARTIALLY_PROCESSED: cannot cancel partially processed "
            "item (WBSTA=B) for order=%s item=%s",
            action.order_id, action.item_no,
        )
        return "BLOCKED_PARTIALLY_PROCESSED"

    # Rule 2: For a quantity change, only the ADDITIONAL units (delta) must be covered
    # by available stock — not the whole new quantity. A decrease frees stock, so its
    # delta is negative and never blocks. new_quantity is already absolute here
    # (resolve_quantity normalized any relative "reduce by N"/"increase by N" upstream),
    # so the delta is computed identically regardless of how the email phrased it.
    if action.action_type == "CHANGE_QTY" and action.new_quantity is not None:
        current_qty = validation_result.get("quantity")
        if current_qty is None:
            # Current quantity unknown → can't compute the delta. Fall back to the
            # conservative absolute check so we never under-reserve stock.
            additional_needed = float(action.new_quantity)
        else:
            additional_needed = action.new_quantity - float(current_qty)

        if additional_needed > available_stock:
            logger.warning(
                "[worker_a] BLOCKED_NO_STOCK: new_qty=%d current=%s additional_needed=%.2f "
                "available=%.2f for order=%s item=%s",
                action.new_quantity, current_qty, additional_needed, available_stock,
                action.order_id, action.item_no,
            )
            return "BLOCKED_NO_STOCK"

    # Rule 3: CHANGE_ADDR — always passes business rules (address validity is human's responsibility)
    if action.action_type == "CHANGE_ADDR":
        logger.info("[worker_a] CHANGE_ADDR: forwarding to human approval.")
        return None

    logger.info("[worker_a] Business rules passed.")
    return None


# ---------------------------------------------------------------------------
# ③-b Relative quantity resolution
# ---------------------------------------------------------------------------

def resolve_quantity(action: ERPActionRequest, validation_result: dict | None) -> str | None:
    """
    For CHANGE_QTY with a RELATIVE quantity_change (e.g. "reduce by 50" → -50),
    resolve it into an absolute new_quantity using the current DB quantity (KWMENG),
    then write the result back into action.new_quantity.

    Must run AFTER run_validation_query (needs the current quantity) and BEFORE
    check_business_rules (which validates the absolute new_quantity against stock).

    Returns:
        None                   — nothing to resolve, or resolved successfully
        "BLOCKED_NO_DATA"      — relative change requested but current quantity unknown
        "BLOCKED_INVALID_QTY"  — resolved quantity would be <= 0
    """
    if action.action_type != "CHANGE_QTY" or action.quantity_change is None:
        return None  # absolute path or not a qty change — nothing to do

    current = validation_result.get("quantity") if validation_result else None
    if current is None:
        logger.warning("[worker_a] Cannot resolve relative quantity — current quantity unknown.")
        return "BLOCKED_NO_DATA"

    resolved = int(current) + action.quantity_change  # quantity_change is signed
    logger.info(
        "[worker_a] Relative qty resolved: current=%s change=%+d → new_quantity=%d",
        current, action.quantity_change, resolved,
    )

    if resolved <= 0:
        logger.warning(
            "[worker_a] BLOCKED_INVALID_QTY: current=%s change=%+d → %d (must be > 0)",
            current, action.quantity_change, resolved,
        )
        return "BLOCKED_INVALID_QTY"

    action.new_quantity = resolved
    action.quantity_change = None  # consumed — downstream uses the absolute value
    return None


# ---------------------------------------------------------------------------
# ②-context — Text-to-SQL 요청 컨텍스트
# ---------------------------------------------------------------------------

def _build_request_context(action: ERPActionRequest) -> str:
    """text2sql에 넘길 요청 컨텍스트 문자열을 만든다.

    LLM이 이 액션을 검증하려면 (필수 5개 컬럼 외에) 어떤 추가 컬럼이 필요한지 판단할
    수 있도록 액션 의도를 서술한다. 필수 5개 컬럼은 컨텍스트와 무관하게 항상 반환된다.
    주의: 현재 check_business_rules는 결정론을 위해 고정 5개 필드만 소비하므로, 추가로
    조회된 컬럼은 아직 다운스트림에서 사용되지 않는다(향후 가변 검증 확장 대비).
    """
    at = action.action_type
    if at == "CHANGE_QTY":
        if action.new_quantity is not None:
            tgt = f"to an absolute quantity of {action.new_quantity}"
        elif action.quantity_change is not None:
            verb = "increase" if action.quantity_change > 0 else "decrease"
            tgt = f"by a relative {verb} of {abs(action.quantity_change)}"
        else:
            tgt = "(quantity change)"
        return (f"- Action: CHANGE_QTY {tgt}. To judge stock feasibility, the current "
                f"ordered quantity and the available stock at the order's own plant matter.")
    if at == "CHANGE_DATE":
        return (f"- Action: CHANGE_DATE to {action.new_date or '(unspecified)'}. "
                f"The current delivery/requested schedule date is relevant for validation.")
    if at == "CANCEL_ITEM":
        return ("- Action: CANCEL_ITEM (cancel this order line). Goods-movement status and "
                "any already-processed/delivered quantity are relevant to whether cancellation is allowed.")
    if at == "CHANGE_ADDR":
        return "- Action: CHANGE_ADDR (change ship-to address). No extra stock/schedule context is needed."
    return f"- Action: {at} (general status validation)."


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
    email_context: dict | None = state.get("email_context")
    errors: list[str] = list(state.get("error_messages", []))

    logger.info("[worker_a] ═══════════════ Worker A START ═══════════════")

    # ── ① Extract ERP action parameters ────────────────────────────────────
    action = extract_erp_action(user_input, email_context=email_context)

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

    # ── OTHER 액션은 시스템이 자동 처리할 수 없다 → 수동 처리로 분기 ────────────────
    # 지원 액션(CHANGE_QTY/CHANGE_DATE/CANCEL_ITEM/CHANGE_ADDR)이 아니므로 검증·OData·
    # 승인 큐를 태우지 않고 MANUAL_REQUIRED로 종료한다. SUCCESS/PENDING_APPROVAL이 되어
    # "완료했다"고 답하면 안 되며(golden도 '수동 처리 필요'로 렌더), 불필요한 OData 호출도 막는다.
    if action.action_type == "OTHER":
        logger.info("[worker_a] action_type=OTHER — 자동 처리 불가, MANUAL_REQUIRED로 분기.")
        logger.info("[worker_a] ═══════════════ Worker A END ═══════════════")
        return {
            "erp_action":              action.model_dump(),
            "erp_validation_result":   None,
            "erp_action_status":       "MANUAL_REQUIRED",
            "odata_response":          None,
            "requires_human_approval": False,
            "error_messages":          errors,
        }

    # ── ② Text-to-SQL validation query ─────────────────────────────────────
    logger.info("[worker_a] ② Running Text-to-SQL validation query …")
    validation_result = run_validation_query(
        action.order_id, action.item_no,
        request_context=_build_request_context(action),
    )
    logger.info("[worker_a] Validation result: %s", validation_result)

    # ── ③ Business rule checks ──────────────────────────────────────────────
    # Resolve relative quantity ("reduce by 50") → absolute, then validate.
    block_reason = resolve_quantity(action, validation_result) \
        or check_business_rules(action, validation_result)

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
