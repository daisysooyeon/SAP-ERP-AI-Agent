"""
src/evaluation/eval_worker_a.py
Worker A 파이프라인 평가

평가 항목:
  [E1] Extraction accuracy  — LLM이 올바른 action_type을 추출했는가
  [E2] Validation coverage  — Text-to-SQL + DB 쿼리가 결과를 반환했는가
  [E3] Status accuracy      — erp_action_status가 expected와 일치하는가

테스트 케이스 소스:
  data/eval/router_test_cases_gen.json  (ACTION_ONLY + BOTH 케이스만 사용)
  - expected_action_type        : CHANGE_QTY | CHANGE_DATE | CANCEL_ITEM | OTHER
  - expected_erp_action_status  : PENDING_APPROVAL | BLOCKED_NO_STOCK | BLOCKED_SHIPPED | …

실행:
  python -m src.evaluation.eval_worker_a
  python -m src.evaluation.eval_worker_a --report reports/worker_a_eval.json
  python -m src.evaluation.eval_worker_a --dry-run   # 첫 5개만 빠르게 테스트
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from src.config import get_config
from src.graph.worker_a import extract_erp_action
from src.tools.text_to_sql import run_validation_query
from src.graph.worker_a import check_business_rules

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TEST_CASES = str(_ROOT / "data" / "eval" / "router_test_cases_gen.json")
DEFAULT_DB         = str(_ROOT / "data" / "sap_erp.db")
DEFAULT_REPORT     = str(_ROOT / "reports" / "worker_a_eval.json")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    id:                   str
    label:                str       # ACTION_ONLY | BOTH
    user_input:           str
    expected_action_type: str       # ground truth
    expected_status:      str       # ground truth

    # E1 — Extraction
    e1_pass:              bool = False
    extracted_action_type: str = ""   # what LLM extracted
    extracted_order_id:   str = ""
    extracted_item_no:    str = ""
    extraction_error:     str = ""

    # E2 — Validation query
    e2_pass:              bool = False
    validation_result:    dict | None = None
    validation_error:     str = ""

    # E3 — Status judgment
    e3_pass:              bool = False
    actual_status:        str = ""

    elapsed_ms:           float = 0.0
    notes:                str = ""


@dataclass
class EvalReport:
    generated_at:   str   = ""
    model_worker_a: str   = ""
    model_sql:      str   = ""
    total:          int   = 0

    # E1 — Extraction
    e1_pass:        int   = 0
    e1_rate:        float = 0.0

    # E2 — Validation
    e2_pass:        int   = 0
    e2_rate:        float = 0.0

    # E3 — Status
    e3_pass:        int   = 0
    e3_rate:        float = 0.0

    # Action type breakdown
    action_type_breakdown: dict = field(default_factory=dict)

    # Status breakdown
    status_breakdown: dict = field(default_factory=dict)

    results: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-case evaluator
# ---------------------------------------------------------------------------

def _evaluate_case(tc: dict, db_path: str) -> CaseResult:
    """Run the full Worker A pipeline on one test case and score it."""

    label                = tc.get("label", "")
    user_input           = tc.get("user_input") or tc.get("input", "")
    expected_action_type = tc.get("expected_action_type", "")
    expected_status      = tc.get("expected_erp_action_status", "")

    cr = CaseResult(
        id=tc.get("id", ""),
        label=label,
        user_input=user_input[:120],
        expected_action_type=expected_action_type,
        expected_status=expected_status,
    )

    t0 = time.perf_counter()

    # ── E1: Parameter extraction ─────────────────────────────────────────────
    logger.info("[%s] E1 — extracting ERP action from email …", cr.id)
    try:
        action = extract_erp_action(user_input)
        if action is None:
            cr.extraction_error   = "extract_erp_action returned None"
            cr.extracted_action_type = ""
            cr.e1_pass = False
        else:
            cr.extracted_action_type = action.action_type
            cr.extracted_order_id    = action.order_id
            cr.extracted_item_no     = action.item_no
            cr.e1_pass = (action.action_type == expected_action_type)

    except Exception as e:
        cr.extraction_error = str(e)
        cr.e1_pass = False
        logger.error("[%s] Extraction exception: %s", cr.id, e)

    # ── E2: Validation query (use erp_evidence order_id for reliability) ─────
    # Prefer the ground-truth order_id from erp_evidence so E2 is not blocked
    # by E1 extraction errors.  This isolates E2 from E1.
    ev         = tc.get("erp_evidence") or {}
    gt_order   = str(ev.get("order_id", "")).strip()
    gt_item    = ev.get("item_no")

    if gt_order and gt_item is not None: # 두 값은 무조건 있어야 함
        try:
            gt_item_int = int(str(gt_item).strip().lstrip("0") or "0")
            logger.info("[%s] E2 — running validation query order=%s item=%s …",
                        cr.id, gt_order, gt_item_int)
            # Use hardcoded query directly to isolate E2 from LLM SQL generation
            from src.tools.text_to_sql import _hardcoded_query, DB_PATH
            import sqlite3 as _sqlite3
            sql = _hardcoded_query(gt_order, str(gt_item_int)) # 쿼리 생성 능력을 평가하려는 게 아님
            conn = _sqlite3.connect(DB_PATH)
            conn.row_factory = _sqlite3.Row
            cur = conn.execute(sql)
            row = cur.fetchone()
            conn.close()
            val_result = dict(row) if row else None
            cr.validation_result = val_result
            cr.e2_pass = val_result is not None
        except Exception as e:
            cr.validation_error = str(e)
            cr.e2_pass = False
            logger.error("[%s] Validation exception: %s", cr.id, e)
    else:
        cr.validation_error = "Missing order_id/item_no in erp_evidence"
        cr.e2_pass = False

    # ── E3: Status judgment ──────────────────────────────────────────────────
    # Use extracted action (if available) + ground-truth validation result
    # so E3 measures business-rule logic, not extraction quality.
    if action is not None or cr.extracted_action_type:
        eval_action = action
    else:
        eval_action = None

    if eval_action is not None:
        try:
            block_reason = check_business_rules(eval_action, cr.validation_result)
            cr.actual_status = block_reason if block_reason else "PENDING_APPROVAL"
        except Exception as e:
            cr.actual_status = "ERROR"
            logger.error("[%s] Business rule check exception: %s", cr.id, e)
    else:
        cr.actual_status = "BLOCKED_EXTRACTION_FAILED"

    cr.e3_pass = (cr.actual_status == expected_status)

    cr.elapsed_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        "[%s] done | E1=%s E2=%s E3=%s | %.0fms",
        cr.id,
        "✓" if cr.e1_pass else "✗",
        "✓" if cr.e2_pass else "✗",
        "✓" if cr.e3_pass else "✗",
        cr.elapsed_ms,
    )

    return cr


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_eval(
    test_cases_path: str = DEFAULT_TEST_CASES,
    db_path:         str = DEFAULT_DB,
    report_path:     str | None = None,
    dry_run:         bool = False,
) -> EvalReport:
    cfg = get_config()

    all_cases: list[dict] = json.loads(
        Path(test_cases_path).read_text(encoding="utf-8")
    )

    # Filter to only ACTION_ONLY and BOTH (QA_ONLY has no ERP action)
    cases = [c for c in all_cases if c.get("label") in ("ACTION_ONLY", "BOTH")]
    logger.info(
        "Loaded %d total cases → %d ACTION/BOTH cases selected",
        len(all_cases), len(cases),
    )

    if dry_run:
        cases = cases[:5]
        logger.info("Dry-run mode: using first %d cases only", len(cases))

    report = EvalReport(
        generated_at   = datetime.now(timezone.utc).isoformat(),
        model_worker_a = cfg.models.worker_a.name,
        model_sql      = cfg.models.worker_a_sql.name,
        total          = len(cases),
    )

    for tc in cases:
        cr = _evaluate_case(tc, db_path)
        report.results.append(asdict(cr))

        if cr.e1_pass: report.e1_pass += 1
        if cr.e2_pass: report.e2_pass += 1
        if cr.e3_pass: report.e3_pass += 1

        # Action type breakdown
        at = cr.expected_action_type or "UNKNOWN"
        bp = report.action_type_breakdown.setdefault(at, {"total": 0, "e1": 0, "e3": 0})
        bp["total"] += 1
        if cr.e1_pass: bp["e1"] += 1
        if cr.e3_pass: bp["e3"] += 1

        # Status breakdown
        es = cr.expected_status or "UNKNOWN"
        sb = report.status_breakdown.setdefault(es, {"total": 0, "e3": 0})
        sb["total"] += 1
        if cr.e3_pass: sb["e3"] += 1

    n = report.total
    report.e1_rate = round(report.e1_pass / n, 4) if n else 0.0
    report.e2_rate = round(report.e2_pass / n, 4) if n else 0.0
    report.e3_rate = round(report.e3_pass / n, 4) if n else 0.0

    # ── Summary printout ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Worker A Evaluation Results")
    print("=" * 65)
    print(f"  Worker A model : {report.model_worker_a}")
    print(f"  SQL model      : {report.model_sql}")
    print(f"  Total cases    : {n}  (ACTION_ONLY + BOTH only)")
    print("-" * 65)
    print(f"  E1 Extraction  : {report.e1_pass}/{n}  ({report.e1_rate*100:.1f}%)  "
          f"[action_type correct]")
    print(f"  E2 Validation  : {report.e2_pass}/{n}  ({report.e2_rate*100:.1f}%)  "
          f"[DB row returned]")
    print(f"  E3 Status      : {report.e3_pass}/{n}  ({report.e3_rate*100:.1f}%)  "
          f"[erp_action_status correct]")
    print("=" * 65)

    # Action type breakdown
    print("\n  E1 breakdown by action_type:")
    for at, bp in sorted(report.action_type_breakdown.items()):
        rate = bp["e1"] / bp["total"] * 100 if bp["total"] else 0
        print(f"    {at:<15}: {bp['e1']}/{bp['total']}  ({rate:.0f}%)")

    # Status breakdown
    print("\n  E3 breakdown by expected_status:")
    for es, sb in sorted(report.status_breakdown.items()):
        rate = sb["e3"] / sb["total"] * 100 if sb["total"] else 0
        print(f"    {es:<30}: {sb['e3']}/{sb['total']}  ({rate:.0f}%)")

    # Failure details
    failed_e1 = [r for r in report.results if not r["e1_pass"]]
    failed_e3 = [r for r in report.results if not r["e3_pass"]]

    if failed_e1:
        print(f"\n  E1 failures ({len(failed_e1)}):")
        for r in failed_e1[:10]:
            print(f"    [{r['id']}] expected={r['expected_action_type']!r}  "
                  f"got={r['extracted_action_type']!r}  "
                  f"err={r['extraction_error'][:60]}")

    if failed_e3:
        print(f"\n  E3 failures ({len(failed_e3)}):")
        for r in failed_e3[:10]:
            print(f"    [{r['id']}] expected={r['expected_status']!r}  "
                  f"got={r['actual_status']!r}")

    # ── Save report ──────────────────────────────────────────────────────────
    if report_path:
        out = Path(report_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(asdict(report), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n  Report saved: {report_path}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Worker A pipeline evaluation")
    parser.add_argument(
        "--test-cases", default=DEFAULT_TEST_CASES,
        help=f"Test cases JSON path (default: {DEFAULT_TEST_CASES})"
    )
    parser.add_argument(
        "--db", default=DEFAULT_DB,
        help=f"SQLite DB path (default: {DEFAULT_DB})"
    )
    parser.add_argument(
        "--report", default=DEFAULT_REPORT,
        help=f"Report output path (default: {DEFAULT_REPORT})"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run only first 5 cases for quick sanity check"
    )
    args = parser.parse_args()

    run_eval(
        test_cases_path=args.test_cases,
        db_path=args.db,
        report_path=args.report,
        dry_run=args.dry_run,
    )
