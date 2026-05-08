"""
src/data/enrich_worker_a_labels.py
────────────────────────────────────────────────────────────────────────────
Enrich router_test_cases_gen.json with Worker A evaluation fields.

Two-phase process:
  Phase 0 (PATCH)  : Replace invalid erp_evidence order_ids with real ones
                     drawn from text2sql_test_cases_gen.json.
  Phase 1 (ENRICH) : Add expected_action_type + expected_erp_action_status
                     derived from the (now-valid) erp_evidence + DB query.

Added fields per case:
  - erp_evidence.order_id      : actual VBELN from DB
  - erp_evidence.item_no       : integer POSNR
  - erp_evidence.item_no_str   : 6-digit zero-padded POSNR string
  - erp_evidence.expected_values : DB snapshot (from text2sql pool)
  - expected_action_type       : CHANGE_QTY | CHANGE_DATE | CANCEL_ITEM | OTHER
  - expected_erp_action_status : PENDING_APPROVAL | BLOCKED_SHIPPED |
                                 BLOCKED_NO_STOCK | BLOCKED_NO_DATA

Run:
  python -m src.data.enrich_worker_a_labels
  python -m src.data.enrich_worker_a_labels --sql-pool data/eval/text2sql_test_cases_gen.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sqlite3
import sys
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

_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT    = str(_ROOT / "data" / "eval" / "router_test_cases_gen.json")
DEFAULT_OUTPUT   = DEFAULT_INPUT
DEFAULT_DB       = str(_ROOT / "data" / "sap_erp.db")
DEFAULT_SQL_POOL = str(_ROOT / "data" / "eval" / "text2sql_test_cases_gen.json")

# ---------------------------------------------------------------------------
# Phase 0 — Load valid order pool from text2sql cases
# ---------------------------------------------------------------------------

def _load_valid_sql_pool(sql_pool_path: str) -> list[dict]:
    """
    Load text2sql test cases that have a real order_id (exclude negative case).
    Each entry has: order_id, item_no, description, expected_values.
    """
    p = Path(sql_pool_path)
    if not p.exists():
        logger.warning("SQL pool file not found: %s — skipping patch phase", sql_pool_path)
        return []
    all_cases = json.loads(p.read_text(encoding="utf-8"))
    # Exclude negative case (9999999999) and cases with no expected_values
    valid = [
        c for c in all_cases
        if c.get("order_id") and str(c["order_id"]) != "9999999999"
    ]
    logger.info("Loaded %d valid SQL pool entries from %s", len(valid), sql_pool_path)
    return valid


def _is_order_valid(order_id: str, db_path: str) -> bool:
    """Quick DB check: does this order_id exist in VBAP?"""
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.execute("SELECT 1 FROM VBAP WHERE VBELN = ? LIMIT 1", (order_id,))
        found = cur.fetchone() is not None
        conn.close()
        return found
    except Exception:
        return False


def _patch_erp_evidence(cases: list[dict], sql_pool: list[dict], db_path: str, seed: int = 42) -> int:
    """
    Phase 0: For each ACTION_ONLY / BOTH case whose erp_evidence.order_id is
    NOT found in the DB, replace it with a randomly chosen valid entry from sql_pool.

    Returns the number of cases patched.
    """
    if not sql_pool:
        return 0

    rng = random.Random(seed)
    patched = 0

    for case in cases:
        if case.get("label") == "QA_ONLY":
            continue

        ev = case.get("erp_evidence") or {}
        raw_oid = str(ev.get("order_id", "")).strip()

        # Check if the current order_id is already valid in DB
        if raw_oid and _is_order_valid(raw_oid, db_path):
            continue  # Already valid — no patch needed

        # Pick a random replacement from the sql pool
        replacement = rng.choice(sql_pool)

        old_oid = raw_oid or "(empty)"
        new_oid = replacement["order_id"]
        new_item = replacement["item_no"]

        # Patch erp_evidence — keep action/description from original
        ev["order_id"]       = new_oid
        ev["item_no"]        = new_item
        ev["expected_values"] = replacement.get("expected_values")
        ev["description"]    = replacement.get("description", ev.get("description", ""))
        case["erp_evidence"] = ev

        logger.info(
            "[PATCH] [%s] order_id %s → %s (item_no=%s)",
            case.get("id"), old_oid, new_oid, new_item,
        )
        patched += 1

    return patched


# ---------------------------------------------------------------------------
# Phase 1 — Enrich with expected_action_type + expected_erp_action_status
# ---------------------------------------------------------------------------

# ② action_type keyword mapping
_QTY_PATTERNS    = re.compile(r"quantit|qty|units?|reduce\s+the\s+order", re.I)
_DATE_PATTERNS   = re.compile(r"deliver(y)?\s+date|reschedule", re.I)
_CANCEL_PATTERNS = re.compile(r"cancel", re.I)


def _map_action_type(action_description: str | None) -> str:
    if not action_description:
        return "OTHER"
    desc = action_description.strip()
    if _CANCEL_PATTERNS.search(desc):
        return "CANCEL_ITEM"
    if _QTY_PATTERNS.search(desc):
        return "CHANGE_QTY"
    if _DATE_PATTERNS.search(desc):
        return "CHANGE_DATE"
    return "OTHER"


def _extract_qty_from_description(desc: str | None) -> int | None:
    if not desc:
        return None
    m = re.search(r"(\d+)\s*units?", desc, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"by\s+(\d+)", desc, re.I)
    if m:
        return int(m.group(1))
    return None


# ③ DB query
_VALIDATION_SQL = """
SELECT
    vbup.WBSTA              AS delivery_status,
    COALESCE(mard.LABST, 0) AS available_stock
FROM VBAP vbap
LEFT JOIN VBUP vbup ON vbap.VBELN = vbup.VBELN AND vbap.POSNR = vbup.POSNR
LEFT JOIN MARD mard ON vbap.MATNR = mard.MATNR
WHERE vbap.VBELN = ? AND vbap.POSNR = ?
LIMIT 1
"""

_BLOCKED_STATUSES = {"C"}


def _query_db(db_path: str, order_id: str, item_no: int) -> dict | None:
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(_VALIDATION_SQL, (order_id, item_no))
        row = cur.fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception as e:
        logger.error("DB query failed for order=%s item=%d: %s", order_id, item_no, e)
        return None


# ④ Derive expected status
def _derive_status(action_type: str, db_row: dict | None, action_desc: str | None) -> str:
    if db_row is None:
        return "BLOCKED_NO_DATA"

    delivery_status = db_row.get("delivery_status") or ""
    available_stock = float(db_row.get("available_stock") or 0)

    if delivery_status in _BLOCKED_STATUSES:
        return "BLOCKED_SHIPPED"

    if action_type == "CHANGE_QTY":
        requested_qty = _extract_qty_from_description(action_desc)
        if requested_qty is not None and requested_qty > available_stock:
            return "BLOCKED_NO_STOCK"

    return "PENDING_APPROVAL"


def _enrich_cases(cases: list[dict], db_path: str) -> tuple[int, int]:
    """
    Phase 1: Add expected_action_type + expected_erp_action_status to each case.
    Returns (enriched_count, skipped_count).
    """
    enriched = 0
    skipped  = 0

    for case in cases:
        label = case.get("label", "")

        if label == "QA_ONLY":
            case.setdefault("expected_action_type",       None)
            case.setdefault("expected_erp_action_status", None)
            continue

        ev = case.get("erp_evidence") or {}
        raw_order = ev.get("order_id")
        raw_item  = ev.get("item_no")

        # Normalise item_no to int
        try:
            item_no_int = int(str(raw_item).strip())
            item_no_str = str(item_no_int).zfill(6)
        except (TypeError, ValueError):
            logger.warning("[%s] Invalid item_no=%s — skipping", case.get("id"), raw_item)
            case["expected_action_type"]       = "OTHER"
            case["expected_erp_action_status"] = "BLOCKED_NO_DATA"
            skipped += 1
            continue

        order_id = str(raw_order).strip() if raw_order else None
        if not order_id:
            case["expected_action_type"]       = "OTHER"
            case["expected_erp_action_status"] = "BLOCKED_NO_DATA"
            skipped += 1
            continue

        # Write normalised item_no back
        ev["item_no"]     = item_no_int
        ev["item_no_str"] = item_no_str
        case["erp_evidence"] = ev

        # ② action_type
        action_desc = ev.get("action") or case.get("action_description") or ""
        action_type = _map_action_type(action_desc)
        case["expected_action_type"] = action_type

        # ③ DB query
        db_row = _query_db(db_path, order_id, item_no_int)

        # ④ Derive status
        status = _derive_status(action_type, db_row, action_desc)
        case["expected_erp_action_status"] = status

        logger.info(
            "[%s] order=%s item=%s action_type=%-12s → %s",
            case.get("id"), order_id, item_no_str, action_type, status,
        )
        enriched += 1

    return enriched, skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def enrich(
    input_path:    str = DEFAULT_INPUT,
    output_path:   str = DEFAULT_OUTPUT,
    db_path:       str = DEFAULT_DB,
    sql_pool_path: str = DEFAULT_SQL_POOL,
    seed:          int = 42,
) -> list[dict]:
    cases: list[dict] = json.loads(
        Path(input_path).read_text(encoding="utf-8")
    )
    logger.info("Loaded %d cases from %s", len(cases), input_path)

    if not Path(db_path).exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    # ── Phase 0: Patch invalid order_ids ────────────────────────────────────
    sql_pool = _load_valid_sql_pool(sql_pool_path)
    patched  = _patch_erp_evidence(cases, sql_pool, db_path, seed=seed)
    logger.info("Phase 0 complete: %d cases patched with valid order_ids", patched)

    # ── Phase 1: Enrich with expected labels ─────────────────────────────────
    enriched, skipped = _enrich_cases(cases, db_path)

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")

    from collections import Counter
    status_counts = Counter(
        c.get("expected_erp_action_status")
        for c in cases
        if c.get("label") != "QA_ONLY"
    )

    print("\n" + "=" * 60)
    print("  Worker A Label Enrichment Results")
    print("=" * 60)
    print(f"  Total cases      : {len(cases)}")
    print(f"  Patched (Phase 0): {patched}")
    print(f"  Enriched         : {enriched}")
    print(f"  Skipped          : {skipped}")
    print(f"  Output file      : {output_path}")
    print("=" * 60)
    print("\n  Expected status distribution (ACTION/BOTH only):")
    for status, cnt in sorted(status_counts.items()):
        print(f"    {status:<30}: {cnt}")

    return cases


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich router_test_cases_gen.json with Worker A evaluation labels"
    )
    parser.add_argument("--input",    default=DEFAULT_INPUT,    help="Input JSON path")
    parser.add_argument("--output",   default=DEFAULT_OUTPUT,   help="Output JSON path (default: in-place)")
    parser.add_argument("--db",       default=DEFAULT_DB,       help="SQLite DB path")
    parser.add_argument("--sql-pool", default=DEFAULT_SQL_POOL, help="text2sql test cases JSON path")
    parser.add_argument("--seed",     type=int, default=42,     help="Random seed for patching")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    enrich(
        input_path=args.input,
        output_path=args.output,
        db_path=args.db,
        sql_pool_path=args.sql_pool,
        seed=args.seed,
    )
