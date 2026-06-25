"""
src/data/generate_golden_responses.py
Generate golden_response for each test case using an LLM with ground-truth context.

Three separate prompts per label to avoid context leakage:
  QA_ONLY     : no ERP context — only qa_question + rag_evidence
  ACTION_ONLY : no user_input — only structured erp_evidence to prevent hallucination
  BOTH        : structured ERP action + RAG answer

Run:
  python -m src.data.generate_golden_responses
  python -m src.data.generate_golden_responses --skip-existing --delay 1.0
  python -m src.data.generate_golden_responses --model openai/gpt-4o
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_core.prompts import ChatPromptTemplate
from src.data._llm_client import build_llm, invoke_with_retry

_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PATH = str(_ROOT / "data" / "eval" / "router_test_cases_gen.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared system prompt (Samsung SDS email template)
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are a customer service agent for the Samsung SDS Logistics Team, composing
professional B2B email replies in English.

STRICTLY follow this template — only the body paragraphs change per case:

Dear Customer,

Hello,

This is the Samsung SDS Logistics Team.

After our internal review of the matter you raised in your email,
[body — 1 to 4 short paragraphs]

Please feel free to contact us anytime for any related inquiries.

Thank you.

Rules:
- Body must be factual. Use ONLY the information provided in the context.
- Do NOT use technical database terms (e.g. MARD, JOIN, VBAP).
- Do NOT invent order numbers, quantities, or dates not given.
- Keep the entire email under 220 words.
- Output the full email only — no explanation, no markdown.
"""

# ---------------------------------------------------------------------------
# Label-specific human prompts
# ---------------------------------------------------------------------------

_QA_HUMAN = """\
The customer asked a knowledge question about SAP ERP.

Question  : {qa_question}
Source    : {rag_evidence}

Write the golden response. Answer the question using the source text.
Do NOT mention any ERP order processing — this is a knowledge-only reply.
"""

_ACTION_HUMAN = """\
The customer requested an ERP order modification. Use ONLY the details below.

Order ID    : {order_id}
Item No     : {item_no}
Action Type : {action_type}
Requested   : {action_desc}
Result      : {erp_status_display}

Write the golden response informing the customer of the result.
Use ONLY the order/item numbers listed above — do not reference any other numbers.
"""

_BOTH_HUMAN = """\
The customer made an ERP order request AND asked a knowledge question.

--- ERP Action ---
Order ID    : {order_id}
Item No     : {item_no}
Action Type : {action_type}
Requested   : {action_desc}
Result      : {erp_status_display}

--- Knowledge Question ---
Question  : {qa_question}
Source    : {rag_evidence}

Write the golden response covering both parts:
1. Inform about the ERP action result (use only the order/item numbers above).
2. Answer the knowledge question using the source text.
"""

_QA_PROMPT     = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", _QA_HUMAN)])
_ACTION_PROMPT = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", _ACTION_HUMAN)])
_BOTH_PROMPT   = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", _BOTH_HUMAN)])

# ---------------------------------------------------------------------------
# ERP status display strings
# PENDING_APPROVAL → SUCCESS (HITL would approve in production)
# ---------------------------------------------------------------------------

_STATUS_DISPLAY: dict[str, str] = {
    "SUCCESS":                   "Action completed successfully.",
    "PENDING_APPROVAL":          "Action completed successfully.",
    "REJECTED":                  "Action rejected by the approver.",
    "FAILED":                    "Action failed due to a system error.",
    "BLOCKED_NO_STOCK":          "BLOCKED — insufficient stock for the requested quantity.",
    "BLOCKED_INVALID_QTY":       "BLOCKED — resulting quantity would be zero or negative.",
    "BLOCKED_SHIPPED":           "BLOCKED — item already fully shipped, cannot be modified.",
    "BLOCKED_EXTRACTION_FAILED": "BLOCKED — could not parse order details from the email.",
    "BLOCKED_NO_DATA":           "BLOCKED — order or item not found in the system.",
    "BLOCKED_VALIDATION":        "BLOCKED — validation error.",
}

_OTHER_STATUS = (
    "This request type requires manual processing by our team "
    "and cannot be handled automatically."
)

# ---------------------------------------------------------------------------
# Derive action_type / erp_status when fields are missing
# ---------------------------------------------------------------------------

def _derive_action_type(action_desc: str | None) -> str:
    if not action_desc:
        return "OTHER"
    if re.search(r"cancel", action_desc, re.I):
        return "CANCEL_ITEM"
    if re.search(r"quantit|qty|units?|reduce\s+the\s+order", action_desc, re.I):
        return "CHANGE_QTY"
    if re.search(r"deliver(y)?\s+date|reschedule", action_desc, re.I):
        return "CHANGE_DATE"
    if re.search(r"address|ship.to|shipping\s+address", action_desc, re.I):
        return "CHANGE_ADDR"
    return "OTHER"


def _derive_erp_status(action_type: str, ev: dict | None, action_desc: str | None) -> str:
    if action_type == "OTHER":
        return "MANUAL_REQUIRED"   # 자동 처리 불가 액션 — SUCCESS/PENDING이 되면 안 됨
    if not ev:
        return "BLOCKED_NO_DATA"
    expected_values = ev.get("expected_values") or {}
    delivery_status = expected_values.get("delivery_status") or ""
    available_stock = float(expected_values.get("available_stock") or 0)
    if delivery_status == "C":
        return "BLOCKED_SHIPPED"
    if action_type == "CHANGE_QTY":
        m = (re.search(r"(\d+)\s*units?", action_desc or "", re.I) or
             re.search(r"by\s+(\d+)", action_desc or "", re.I))
        if m and int(m.group(1)) > available_stock:
            return "BLOCKED_NO_STOCK"
    return "PENDING_APPROVAL"


def _status_display(erp_status: str, action_type: str) -> str:
    if action_type == "OTHER":
        return _OTHER_STATUS
    return _STATUS_DISPLAY.get(erp_status, f"Status: {erp_status}")


def _rag_text(case: dict) -> str:
    evidence = case.get("rag_evidence") or ""
    if isinstance(evidence, list):
        evidence = "\n\n---\n\n".join(evidence)
    return str(evidence)[:1500]

# ---------------------------------------------------------------------------
# Build label-specific LLM inputs
# ---------------------------------------------------------------------------

def _build_inputs(case: dict) -> tuple[ChatPromptTemplate, dict]:
    label       = case.get("label", "QA_ONLY")
    ev          = case.get("erp_evidence") or {}
    action_desc = ev.get("action") or ""

    action_type = case.get("expected_action_type") or _derive_action_type(action_desc)
    erp_status  = (
        case.get("expected_erp_action_status")
        or _derive_erp_status(action_type, ev, action_desc)
    ) if label != "QA_ONLY" else "N/A"

    # Save derived values back (so JSON reflects the ground truth)
    if not case.get("expected_action_type") and label != "QA_ONLY":
        case["expected_action_type"] = action_type
    if not case.get("expected_erp_action_status") and label != "QA_ONLY":
        case["expected_erp_action_status"] = erp_status

    status_disp = _status_display(erp_status, action_type)

    if label == "QA_ONLY":
        return _QA_PROMPT, {
            "qa_question":  case.get("qa_question") or "",
            "rag_evidence": _rag_text(case),
        }

    order_id   = str(ev.get("order_id", ""))
    item_no    = str(ev.get("item_no", ""))

    if label == "ACTION_ONLY":
        return _ACTION_PROMPT, {
            "order_id":          order_id,
            "item_no":           item_no,
            "action_type":       action_type,
            "action_desc":       action_desc,
            "erp_status_display": status_disp,
        }

    # BOTH
    return _BOTH_PROMPT, {
        "order_id":          order_id,
        "item_no":           item_no,
        "action_type":       action_type,
        "action_desc":       action_desc,
        "erp_status_display": status_disp,
        "qa_question":       case.get("qa_question") or "",
        "rag_evidence":      _rag_text(case),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_golden_responses(
    path: str = DEFAULT_PATH,
    skip_existing: bool = False,
    delay: float = 1.5,
    model_name: str | None = None,
) -> None:
    data_path = Path(path)
    with open(data_path, encoding="utf-8") as f:
        dataset: list[dict] = json.load(f)

    llm = build_llm(model_name=model_name, temperature=0.0)
    updated = skipped = failed = 0

    for i, case in enumerate(dataset):
        cid   = case.get("id", f"#{i}")
        label = case.get("label", "?")

        if skip_existing and case.get("golden_response"):
            logger.info("[%s] skipped", cid)
            skipped += 1
            continue

        prompt, inputs = _build_inputs(case)
        chain = prompt | llm

        logger.info(
            "[%d/%d] %s  label=%-11s  erp_status=%-35s  action_type=%s",
            i + 1, len(dataset), cid, label,
            case.get("expected_erp_action_status") or "N/A",
            case.get("expected_action_type") or "N/A",
        )

        raw = invoke_with_retry(chain, inputs, label=f"golden/{cid}")
        time.sleep(delay)

        if raw is None:
            logger.warning("[%s] generation failed — keeping existing value", cid)
            failed += 1
            continue

        case["golden_response"] = raw.strip()
        updated += 1

    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("  Golden Response Generation Complete")
    print("=" * 60)
    print(f"  Updated : {updated}")
    print(f"  Skipped : {skipped}  (--skip-existing)")
    print(f"  Failed  : {failed}")
    print(f"  Total   : {len(dataset)}")
    print(f"  Output  : {path}")
    print("=" * 60)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate per-case golden_response using LLM with ground-truth context"
    )
    p.add_argument("--input",         default=DEFAULT_PATH, help="Path to test cases JSON")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip cases that already have a golden_response")
    p.add_argument("--delay",         type=float, default=1.5,
                   help="Seconds between API calls (default: 1.5)")
    p.add_argument("--model",         default=None,
                   help="OpenRouter model override (default: models.data_gen.name in configs.yaml)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_golden_responses(
        path=args.input,
        skip_existing=args.skip_existing,
        delay=args.delay,
        model_name=args.model,
    )
