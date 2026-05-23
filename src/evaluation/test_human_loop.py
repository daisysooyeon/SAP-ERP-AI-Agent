"""
src/evaluation/test_human_loop.py
Human-in-the-Loop end-to-end simulation (no server or Slack required)

What this tests:
  1. Full agent run with an ERP action email → graph pauses at PENDING_APPROVAL
  2. Approval scenario  : human_approved=True  → SQLite UPDATE → SUCCESS
  3. Rejection scenario : human_approved=False → no DB change  → REJECTED

Run:
  python -m src.evaluation.test_human_loop
  python -m src.evaluation.test_human_loop --approve   # approve only
  python -m src.evaluation.test_human_loop --reject    # reject only
  python -m src.evaluation.test_human_loop --email "Cancel order 4976 item 10"
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

_ROOT    = Path(__file__).resolve().parent.parent.parent
_DB_PATH = str(_ROOT / "data" / "sap_erp.db")

# Default test email — references a real order in the DB
DEFAULT_EMAIL = (
    "Dear SAP team,\n\n"
    "Please cancel item 10 of sales order 0000015353. "
    "The customer has withdrawn their purchase request.\n\n"
    "Best regards,\nDaisy"
)

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _snapshot_db(order_id: str, item_no: int) -> dict | None:
    """Read current VBAP row for the order/item."""
    try:
        conn = sqlite3.connect(_DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT VBELN, POSNR, KWMENG, ABGRU FROM VBAP "
            "WHERE VBELN = ? AND POSNR = ? LIMIT 1",
            (order_id, item_no),
        )
        row = cur.fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception as e:
        logger.error("DB snapshot failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

def _run_scenario(
    email: str,
    approve: bool,
    scenario_label: str,
) -> dict:
    """
    Run a single approve/reject scenario and return a result summary dict.
    """
    from src.main import graph  # imported here to avoid import at module load

    thread_id = str(uuid.uuid4())
    config    = {"configurable": {"thread_id": thread_id}}

    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  Scenario : {scenario_label}")
    print(f"  Thread   : {thread_id}")
    print(f"  Email    : {email[:80]}{'…' if len(email) > 80 else ''}")
    print(sep)

    # ── Step 1: Run agent ────────────────────────────────────────────────────
    print("\n[Step 1] Running agent …")
    result = graph.invoke(
        {
            "user_input":             email,
            "error_messages":         [],
            "requires_human_approval": False,
            "human_approved":         None,
        },
        config=config,
    )

    status_after_run = result.get("erp_action_status")
    erp_action       = result.get("erp_action")
    requires_approval = result.get("requires_human_approval", False)

    print(f"  erp_action_status    : {status_after_run}")
    print(f"  requires_approval    : {requires_approval}")
    print(f"  erp_action           : {json.dumps(erp_action, ensure_ascii=False, indent=4)}")

    summary = {
        "scenario":          scenario_label,
        "thread_id":         thread_id,
        "status_after_run":  status_after_run,
        "requires_approval": requires_approval,
        "erp_action":        erp_action,
        "db_before":         None,
        "db_after":          None,
        "final_status":      None,
        "passed":            False,
    }

    # ── Check interrupt ──────────────────────────────────────────────────────
    if status_after_run != "PENDING_APPROVAL":
        print(f"\n  ⚠️  Graph did NOT pause — status is '{status_after_run}', not PENDING_APPROVAL.")
        print(    "     This may be a BLOCKED_* status (business rule check) or an extraction failure.")
        print(    "     Human-in-the-Loop was not triggered. Nothing to approve/reject.\n")
        summary["final_status"] = status_after_run
        return summary

    print("\n  ✅ Graph paused at PENDING_APPROVAL — interrupt working correctly.")

    # ── DB snapshot before ───────────────────────────────────────────────────
    if erp_action:
        try:
            order_id = erp_action.get("order_id", "")
            item_no  = int(str(erp_action.get("item_no", "0")).lstrip("0") or "0")
            db_before = _snapshot_db(order_id, item_no)
            summary["db_before"] = db_before
            print(f"\n[DB before] {db_before}")
        except Exception as e:
            logger.warning("Could not snapshot DB before: %s", e)

    # ── Step 2: Simulate human decision ─────────────────────────────────────
    from langgraph.types import Command

    decision_label = "APPROVED ✅" if approve else "REJECTED ❌"
    print(f"\n[Step 2] Simulating human decision: {decision_label}")

    for chunk in graph.stream(Command(resume=approve), config=config):
        node_name = list(chunk.keys())[0] if chunk else "?"
        print(f"  → node executed: {node_name}")

    # ── Step 3: Final state ──────────────────────────────────────────────────
    print("\n[Step 3] Fetching final state …")
    final_state  = graph.get_state(config)
    final_status = "UNKNOWN"
    errors: list = []

    if final_state and final_state.values:
        final_status = final_state.values.get("erp_action_status", "UNKNOWN")
        errors       = final_state.values.get("error_messages", [])

    summary["final_status"] = final_status

    print(f"  final erp_action_status : {final_status}")
    if errors:
        print(f"  error_messages          : {errors}")

    # ── DB snapshot after ────────────────────────────────────────────────────
    if erp_action:
        try:
            db_after = _snapshot_db(order_id, item_no)
            summary["db_after"] = db_after
            print(f"\n[DB after ] {db_after}")

            if approve and db_before and db_after:
                if db_before != db_after:
                    print("  ✅ DB row was updated.")
                else:
                    print("  ⚠️  DB row unchanged after approval — check human_loop_node.")
            elif not approve:
                if db_before == db_after:
                    print("  ✅ DB row unchanged after rejection — correct.")
                else:
                    print("  ❌ DB row changed after rejection — unexpected!")
        except Exception as e:
            logger.warning("Could not snapshot DB after: %s", e)

    # ── Pass/fail judgment ───────────────────────────────────────────────────
    expected_status = "SUCCESS" if approve else "REJECTED"
    summary["passed"] = (final_status == expected_status)

    result_icon = "✅ PASS" if summary["passed"] else "❌ FAIL"
    print(f"\n  Expected status : {expected_status}")
    print(f"  Actual status   : {final_status}")
    print(f"  Result          : {result_icon}")

    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_test(email: str, run_approve: bool, run_reject: bool) -> None:
    results = []

    if run_approve:
        results.append(_run_scenario(email, approve=True,  scenario_label="Approve"))
    if run_reject:
        results.append(_run_scenario(email, approve=False, scenario_label="Reject"))

    # ── Final summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Human-in-the-Loop Test Summary")
    print("=" * 60)
    for r in results:
        icon   = "✅ PASS" if r["passed"] else "❌ FAIL"
        status = r.get("final_status", "N/A")
        print(f"  {icon}  [{r['scenario']:<8}]  final_status={status}")
    print("=" * 60)

    all_passed = all(r["passed"] for r in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Human-in-the-Loop end-to-end test (no server required)"
    )
    parser.add_argument(
        "--email", default=DEFAULT_EMAIL,
        help="Email text to process (default: built-in cancel example)",
    )
    parser.add_argument(
        "--approve", action="store_true",
        help="Run approval scenario only",
    )
    parser.add_argument(
        "--reject", action="store_true",
        help="Run rejection scenario only",
    )
    args = parser.parse_args()

    # If neither flag is set, run both scenarios
    run_approve = args.approve or (not args.approve and not args.reject)
    run_reject  = args.reject  or (not args.approve and not args.reject)

    run_test(
        email=args.email,
        run_approve=run_approve,
        run_reject=run_reject,
    )
