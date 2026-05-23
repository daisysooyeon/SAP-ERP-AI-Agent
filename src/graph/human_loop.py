"""
src/graph/human_loop.py
Human-in-the-Loop node

How it fits in the LangGraph flow:
  1. worker_a_node sets erp_action_status="PENDING_APPROVAL" and
     requires_human_approval=True, then the graph pauses at this node
     via interrupt_before=["human_loop_node"].
  2. The FastAPI /api/approve endpoint receives the Slack button callback,
     updates the checkpoint with human_approved=True/False, and resumes
     the graph.
  3. This node is then executed with the updated state:
       - human_approved == True  → apply SQLite UPDATE → SUCCESS
       - human_approved == False → mark REJECTED, no DB write

SQLite UPDATE mapping:
  action_type  | table | column   | value
  ─────────────┼───────┼──────────┼────────────────────────────
  CHANGE_QTY   | VBAP  | KWMENG   | new_quantity (REAL)
  CHANGE_DATE  | VBEP  | EDATU    | new_date (YYYYMMDD text)
  CANCEL_ITEM  | VBAP  | ABGRU    | "ZZ" (rejection reason code)
"""

from __future__ import annotations

import logging
import re
import sqlite3

from langgraph.types import interrupt

from src.config import get_config
from src.graph.state import AgentState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SQLite UPDATE helpers
# ---------------------------------------------------------------------------

def _get_db_path() -> str:
    return get_config().paths.erp_db


def _execute_update(sql: str, params: tuple) -> int:
    """Execute a single UPDATE and return rowcount."""
    db_path = _get_db_path()
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(sql, params)
        conn.commit()
        rowcount = cursor.rowcount
        logger.info("[human_loop] SQL executed — rows affected: %d", rowcount)
        logger.debug("[human_loop] SQL: %s | params: %s", sql.strip(), params)
        return rowcount
    except sqlite3.Error as e:
        conn.rollback()
        logger.error("[human_loop] SQLite error: %s", e)
        raise
    finally:
        conn.close()


def _date_to_sap(date_str: str | None) -> str | None:
    """
    Convert ISO date "YYYY-MM-DD" to SAP EDATU format "YYYYMMDD".
    Returns None if conversion fails.
    """
    if not date_str:
        return None
    # Already YYYYMMDD
    if re.fullmatch(r"\d{8}", date_str):
        return date_str
    # ISO format
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", date_str)
    if m:
        return m.group(1) + m.group(2) + m.group(3)
    logger.warning("[human_loop] Unrecognised date format: %s", date_str)
    return None


def apply_erp_update(erp_action: dict) -> dict:
    """
    Apply the approved ERP action to the SQLite DB.

    Parameters
    ----------
    erp_action : dict  (from ERPActionRequest.model_dump())
        Keys: order_id, item_no, action_type, new_quantity, new_date

    Returns
    -------
    dict with keys:
        success  : bool
        rowcount : int
        message  : str
    """
    order_id    = erp_action.get("order_id", "")
    item_no_raw = erp_action.get("item_no", "")
    action_type = erp_action.get("action_type", "")

    # item_no may arrive as a zero-padded string ("000010") or int (10)
    try:
        item_no = int(str(item_no_raw).lstrip("0") or "0")
    except ValueError:
        item_no = 0

    logger.info(
        "[human_loop] Applying update: order=%s item=%s action=%s",
        order_id, item_no, action_type,
    )

    try:
        if action_type == "CHANGE_QTY":
            new_qty = erp_action.get("new_quantity")
            if new_qty is None:
                return {"success": False, "rowcount": 0,
                        "message": "CHANGE_QTY requires new_quantity"}
            rowcount = _execute_update(
                "UPDATE VBAP SET KWMENG = ? WHERE VBELN = ? AND POSNR = ?",
                (float(new_qty), order_id, item_no),
            )

        elif action_type == "CHANGE_DATE":
            new_date_raw = erp_action.get("new_date")
            new_date_sap = _date_to_sap(new_date_raw)
            if new_date_sap is None:
                return {"success": False, "rowcount": 0,
                        "message": f"CHANGE_DATE: invalid date '{new_date_raw}'"}
            rowcount = _execute_update(
                "UPDATE VBEP SET EDATU = ? WHERE VBELN = ? AND POSNR = ?",
                (new_date_sap, order_id, item_no),
            )

        elif action_type == "CANCEL_ITEM":
            # Set rejection reason code "ZZ" in VBAP
            rowcount = _execute_update(
                "UPDATE VBAP SET ABGRU = ? WHERE VBELN = ? AND POSNR = ?",
                ("ZZ", order_id, item_no),
            )

        elif action_type == "CHANGE_ADDR":
            # For shipping address changes, we only simulate the update
            # because there is no ADRC/VBKD table in the local SQLite mock DB
            logger.info("[human_loop] Simulated CHANGE_ADDR update for order=%s", order_id)
            return {
                "success": True,
                "rowcount": 1,
                "message": "CHANGE_ADDR applied successfully (simulated).",
            }

        else:
            return {
                "success": False,
                "rowcount": 0,
                "message": f"Unknown action_type: '{action_type}'",
            }

    except sqlite3.Error as e:
        return {"success": False, "rowcount": 0, "message": str(e)}

    if rowcount == 0:
        return {
            "success": False,
            "rowcount": 0,
            "message": (
                f"No rows updated for order={order_id} item={item_no}. "
                "Record may not exist in DB."
            ),
        }

    return {
        "success": True,
        "rowcount": rowcount,
        "message": f"{action_type} applied successfully ({rowcount} row(s) updated).",
    }


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def human_loop_node(state: AgentState) -> dict:
    """
    LangGraph Human-in-the-Loop node.

    LangGraph 1.x interrupt() 방식:
      - 노드 실행 중 interrupt()를 호출해 일시 중단
      - 외부에서 Command(resume=True/False)로 재개하면 interrupt() 반환값으로 결과 수신
      - True(승인) → DB UPDATE 실행 → SUCCESS
      - False(거절) → DB 변경 없음 → REJECTED

    Reads:
        state["erp_action"] — ERPActionRequest dict from worker_a

    Returns a partial AgentState update:
        {
            "erp_action_status": "SUCCESS" | "REJECTED" | "FAILED",
            "human_approved":    bool,
            "error_messages":    [...],
        }
    """
    erp_action: dict | None = state.get("erp_action")
    errors: list[str]       = list(state.get("error_messages", []))

    logger.info("[human_loop] ═══════════════ Human Loop START ═══════════════")

    # ── Human decision via interrupt() ──────────────────────────────────────
    human_approved: bool = interrupt({
        "erp_action": erp_action,
        "message":    "Approve or reject this ERP action? (True=Approve, False=Reject)",
    })
    logger.info("[human_loop] human_approved=%s", human_approved)

    # ── Case 1: Rejected ────────────────────────────────────────────────────
    if not human_approved:
        logger.info("[human_loop] Action REJECTED by human reviewer.")
        return {
            "erp_action_status": "REJECTED",
            "human_approved":    False,
            "error_messages":    errors,
        }

    # ── Case 2: Approved — apply DB update ──────────────────────────────────
    if erp_action is None:
        msg = "human_loop: human_approved=True but erp_action is None — cannot update DB."
        logger.error("[human_loop] %s", msg)
        errors.append(msg)
        return {
            "erp_action_status": "FAILED",
            "error_messages":    errors,
        }

    result = apply_erp_update(erp_action)

    if result["success"]:
        logger.info("[human_loop] DB update SUCCESS: %s", result["message"])
        logger.info("[human_loop] ═══════════════ Human Loop END ═══════════════")
        return {
            "erp_action_status": "SUCCESS",
            "human_approved":    True,
            "error_messages":    errors,
        }
    else:
        msg = f"human_loop: DB update FAILED — {result['message']}"
        logger.error("[human_loop] %s", msg)
        errors.append(msg)
        logger.info("[human_loop] ═══════════════ Human Loop END ═══════════════")
        return {
            "erp_action_status": "FAILED",
            "human_approved":    True,
            "error_messages":    errors,
        }
