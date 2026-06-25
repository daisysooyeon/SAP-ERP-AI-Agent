"""
src/evaluation/eval_worker_a.py
Worker A 파이프라인 평가

평가 항목:
  [E1] Extraction accuracy  — LLM이 올바른 action_type을 추출했는가
  [E2] Validation accuracy  — 운영 경로(run_validation_query, LLM text2sql)가 행을 반환하고,
                              그 5개 필드 값이 결정론적 골든(_hardcoded_query)과 일치하는가.
                              비즈니스 룰이 안 읽는 material_name/delivery_date 오류까지 잡는다.
  [E3] Status accuracy      — erp_action_status가 expected와 일치하는가

테스트 케이스 소스:
  data/eval/router_test_cases_gen.json  (ACTION_ONLY + BOTH 케이스만 사용)
  - expected_action_type        : CHANGE_QTY | CHANGE_DATE | CANCEL_ITEM | OTHER
  - expected_erp_action_status  : PENDING_APPROVAL | BLOCKED_NO_STOCK | BLOCKED_SHIPPED | …

실행:
  python -m src.evaluation.eval_worker_a
  python -m src.evaluation.eval_worker_a --report reports/worker_a_eval_result.json
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
from src.graph.worker_a import (
    extract_erp_action,
    resolve_quantity,
    check_business_rules,
    _build_request_context,
)
from src.tools.text2sql import run_validation_query, _hardcoded_query, DB_PATH as _T2S_DB_PATH

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
DEFAULT_REPORT     = str(_ROOT / "reports" / "worker_a_eval_result.json")

# E2 값 일치 검사 대상 — text2sql이 보장해야 하는 5개 alias 컬럼
REQUIRED_FIELDS = ["material_name", "quantity", "delivery_status", "delivery_date", "available_stock"]


def _execute_golden(order_id: str, item_no: int) -> dict | None:
    """결정론적 레퍼런스(_hardcoded_query)를 실행해 정답 행을 만든다.
    운영 경로(run_validation_query)와 같은 DB를 쓰도록 text2sql.DB_PATH로 연결한다."""
    import sqlite3
    conn = sqlite3.connect(_T2S_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(_hardcoded_query(str(order_id), str(item_no))).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def _values_equal(a, b) -> bool:
    """숫자는 부동소수 허용오차로, 그 외는 동등 비교."""
    if a is None and b is None:
        return True
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-6
    return a == b


def _digits_equal(a, b) -> bool:
    """order_id/item_no를 정수로 정규화해 비교 — 패딩('0000017343')·형식 차이를 무시한다.
    어느 한쪽이라도 숫자로 해석 불가(빈값/None)면 False."""
    try:
        na = int("".join(ch for ch in str(a) if ch.isdigit()) or "x")
        nb = int("".join(ch for ch in str(b) if ch.isdigit()) or "x")
        return na == nb
    except ValueError:
        return False


def _compare_to_golden(actual: dict | None, golden: dict | None) -> list[dict]:
    """운영 경로 결과(actual)와 정답(golden)의 5개 필드를 비교, 불일치 목록 반환.
    golden 자체가 없으면(존재하지 않는 주문 등) 비교 대상이 아니므로 빈 리스트."""
    if golden is None:
        return []
    if actual is None:
        return [{"column": "<row>", "expected": "row matching golden", "actual": "no row returned"}]
    mismatches: list[dict] = []
    for col in REQUIRED_FIELDS:
        exp, act = golden.get(col), actual.get(col)
        if not _values_equal(act, exp):
            mismatches.append({"column": col, "expected": exp, "actual": act})
    return mismatches


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
    e1_pass:              bool = False   # action_type 정답 일치
    e1_order_match:       bool = False   # 추출 order_id가 ground-truth와 일치
    e1_item_match:        bool = False   # 추출 item_no가 ground-truth와 일치
    extracted_action_type: str = ""   # what LLM extracted
    extracted_order_id:   str = ""
    extracted_item_no:    str = ""
    extraction_error:     str = ""

    # E2 — Validation query (real production path + value match vs golden)
    e2_pass:              bool = False   # 행 반환 AND 5개 필드 모두 골든과 일치
    e2_row_returned:      bool = False   # 운영 경로가 행을 반환했는가 (coverage)
    e2_values_match:      bool = False   # 반환된 5개 필드 값이 골든과 일치하는가
    validation_result:    dict | None = None  # 운영 경로(run_validation_query) 결과
    golden_result:        dict | None = None  # 결정론적 _hardcoded_query 정답
    validation_mismatches: list = field(default_factory=list)
    validation_error:     str = ""

    # E3 — Status judgment
    e3_pass:              bool = False
    actual_status:        str = ""

    elapsed_ms:           float = 0.0


@dataclass
class EvalReport:
    generated_at:   str   = ""
    model_worker_a: str   = ""
    model_sql:      str   = ""
    total:          int   = 0

    # E1 — Extraction
    e1_pass:        int   = 0   # action_type 정답
    e1_rate:        float = 0.0
    e1_order_pass:  int   = 0   # order_id 추출 일치
    e1_order_rate:  float = 0.0
    e1_item_pass:   int   = 0   # item_no 추출 일치
    e1_item_rate:   float = 0.0
    # action_type/order_id/item_no가 모두 맞은 케이스 (추출 완전 정답)
    e1_full_pass:   int   = 0
    e1_full_rate:   float = 0.0
    # 추출 id가 틀린 케이스 상세
    e1_id_mismatch_cases: list = field(default_factory=list)

    # E2 — Validation (row returned AND values match golden)
    e2_pass:        int   = 0   # 행 반환 AND 값 일치 (최종 E2 통과)
    e2_rate:        float = 0.0
    e2_row_pass:    int   = 0   # 행 반환 (coverage)
    e2_row_rate:    float = 0.0
    e2_values_pass: int   = 0   # 값 일치
    e2_values_rate: float = 0.0
    # 골든과 값이 어긋난 케이스 상세 (text2sql이 행은 줬지만 값이 틀린 경우)
    e2_mismatch_cases: list = field(default_factory=list)

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

def _evaluate_case(tc: dict) -> CaseResult:
    """Run the full Worker A pipeline on one test case and score it.

    DB 경로는 운영 코드(text2sql.DB_PATH)가 결정하므로 별도 인자를 받지 않는다 —
    E2의 운영 경로(run_validation_query)와 골든(_execute_golden)이 같은 DB를 쓴다."""

    label                = tc.get("label", "")
    user_input           = tc.get("user_input", "")
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

    # erp_evidence ground-truth — E1의 order_id/item_no 추출 검증과 E2 입력에 공용 사용
    ev          = tc.get("erp_evidence") or {}
    gt_order    = str(ev.get("order_id", "")).strip()
    gt_item_raw = ev.get("item_no")
    gt_item_int: int | None = None
    if gt_item_raw is not None:
        try:
            gt_item_int = int(str(gt_item_raw).strip().lstrip("0") or "0")
        except ValueError:
            gt_item_int = None

    # ── E1: Parameter extraction ─────────────────────────────────────────────
    # 운영 경로와 동일하게 진입 전 이메일 전처리 결과를 hint로 같이 넘긴다.
    # (전처리 실패 시 email_context.preprocess_ok=False, 다운스트림이 알아서 무시)
    from src.preprocess import preprocess_email
    email_ctx = preprocess_email(user_input).model_dump()

    logger.info("[%s] E1 — extracting ERP action from email …", cr.id)
    try:
        action = extract_erp_action(user_input, email_context=email_ctx)
        if action is None:
            cr.extraction_error   = "extract_erp_action returned None"
            cr.extracted_action_type = ""
            cr.e1_pass = False
        else:
            cr.extracted_action_type = action.action_type
            cr.extracted_order_id    = action.order_id
            cr.extracted_item_no     = action.item_no
            cr.e1_pass = (action.action_type == expected_action_type)
            # order_id/item_no도 올바르게 추출했는지 검증 (패딩 차이는 무시하고 숫자로 비교)
            cr.e1_order_match = _digits_equal(action.order_id, gt_order)
            cr.e1_item_match  = _digits_equal(action.item_no, gt_item_raw)

    except Exception as e:
        cr.extraction_error = str(e)
        cr.e1_pass = False
        logger.error("[%s] Extraction exception: %s", cr.id, e)

    # ── E2: Validation query — REAL production path + value match vs golden ──
    # 운영과 동일하게 run_validation_query(LLM text2sql → 0건이면 hardcoded 재시도)를 태워
    # 평가가 프로덕션을 그대로 반영하게 한다. 정답(golden)은 결정론적 _hardcoded_query 결과.
    # 단순 "행이 나왔나"를 넘어 5개 필드 값 일치까지 보므로, 비즈니스 룰이 읽지 않는
    # material_name/delivery_date 오류도 여기서 잡힌다.
    # 추출(E1)과 분리하기 위해 입력 order_id/item_no는 erp_evidence의 ground-truth를 쓴다.
    # (gt_order/gt_item_int는 함수 상단에서 1회 계산해 E1 검증과 공유한다.)
    if gt_order and gt_item_int is not None: # 두 값은 무조건 있어야 함
        try:
            logger.info("[%s] E2 — running validation query order=%s item=%s …",
                        cr.id, gt_order, gt_item_int)

            # 운영 경로: LLM이 SQL을 생성·실행 (0건이면 내부적으로 hardcoded 재시도)
            # 운영과 동일하게 요청 컨텍스트도 함께 넘긴다(추출된 action이 있을 때).
            req_ctx = _build_request_context(action) if action is not None else None
            val_result = run_validation_query(gt_order, str(gt_item_int), request_context=req_ctx)
            cr.validation_result = val_result
            cr.e2_row_returned   = val_result is not None

            # 정답: 결정론적 하드코딩 쿼리 결과
            golden = _execute_golden(gt_order, gt_item_int)
            cr.golden_result = golden

            # 5개 필드 값 일치 검사
            cr.validation_mismatches = _compare_to_golden(val_result, golden)
            cr.e2_values_match = cr.e2_row_returned and not cr.validation_mismatches
            cr.e2_pass = cr.e2_row_returned and cr.e2_values_match
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
            # 운영(worker_a_node)과 동일하게 상대수량을 먼저 절대값으로 환산한 뒤 룰 검사.
            # 이게 빠지면 "reduce by 50" 같은 상대수량 케이스의 재고/INVALID_QTY 룰이 누락된다.
            if eval_action.action_type == "OTHER":
                # worker_a_node와 동일: OTHER는 자동 처리 불가 → 룰 검사 전에 MANUAL_REQUIRED.
                cr.actual_status = "MANUAL_REQUIRED"
            else:
                block_reason = resolve_quantity(eval_action, cr.validation_result) \
                    or check_business_rules(eval_action, cr.validation_result)
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
        cr = _evaluate_case(tc)
        report.results.append(asdict(cr))

        if cr.e1_pass: report.e1_pass += 1
        if cr.e1_order_match: report.e1_order_pass += 1
        if cr.e1_item_match:  report.e1_item_pass += 1
        if cr.e1_pass and cr.e1_order_match and cr.e1_item_match:
            report.e1_full_pass += 1
        if cr.e2_pass: report.e2_pass += 1
        if cr.e2_row_returned: report.e2_row_pass += 1
        if cr.e2_values_match: report.e2_values_pass += 1
        if cr.e3_pass: report.e3_pass += 1

        # 추출 id(order_id/item_no)가 ground-truth와 어긋난 케이스 기록
        if cr.extracted_action_type and (not cr.e1_order_match or not cr.e1_item_match):
            gt_ev = tc.get("erp_evidence") or {}
            report.e1_id_mismatch_cases.append({
                "id":               cr.id,
                "expected_order":   gt_ev.get("order_id"),
                "extracted_order":  cr.extracted_order_id,
                "order_match":      cr.e1_order_match,
                "expected_item":    gt_ev.get("item_no"),
                "extracted_item":   cr.extracted_item_no,
                "item_match":       cr.e1_item_match,
            })

        # text2sql이 행은 반환했지만 값이 골든과 어긋난 케이스 기록
        if cr.e2_row_returned and cr.validation_mismatches:
            report.e2_mismatch_cases.append({
                "id":            cr.id,
                "order_id":      cr.extracted_order_id or (tc.get("erp_evidence") or {}).get("order_id"),
                "item_no":       cr.extracted_item_no,
                "actual_result": cr.validation_result,
                "golden_result": cr.golden_result,
                "mismatches":    cr.validation_mismatches,
            })

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
    report.e1_order_rate = round(report.e1_order_pass / n, 4) if n else 0.0
    report.e1_item_rate  = round(report.e1_item_pass / n, 4) if n else 0.0
    report.e1_full_rate  = round(report.e1_full_pass / n, 4) if n else 0.0
    report.e2_rate = round(report.e2_pass / n, 4) if n else 0.0
    report.e2_row_rate    = round(report.e2_row_pass / n, 4) if n else 0.0
    report.e2_values_rate = round(report.e2_values_pass / n, 4) if n else 0.0
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
    print(f"     ├─ order_id  : {report.e1_order_pass}/{n}  ({report.e1_order_rate*100:.1f}%)")
    print(f"     ├─ item_no   : {report.e1_item_pass}/{n}  ({report.e1_item_rate*100:.1f}%)")
    print(f"     └─ all 3 ✓   : {report.e1_full_pass}/{n}  ({report.e1_full_rate*100:.1f}%)  "
          f"[action_type + order_id + item_no]")
    print(f"  E2 Validation  : {report.e2_pass}/{n}  ({report.e2_rate*100:.1f}%)  "
          f"[real text2sql: row + values match golden]")
    print(f"     ├─ row returned : {report.e2_row_pass}/{n}  ({report.e2_row_rate*100:.1f}%)")
    print(f"     └─ values match : {report.e2_values_pass}/{n}  ({report.e2_values_rate*100:.1f}%)")
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
        print(f"\n  E1 action_type failures ({len(failed_e1)}):")
        for r in failed_e1[:10]:
            print(f"    [{r['id']}] expected={r['expected_action_type']!r}  "
                  f"got={r['extracted_action_type']!r}  "
                  f"err={r['extraction_error'][:60]}")

    # 추출 id(order_id/item_no) 불일치 상세
    if report.e1_id_mismatch_cases:
        print(f"\n  E1 id mismatches ({len(report.e1_id_mismatch_cases)})  "
              f"[order_id/item_no extracted ≠ ground-truth]:")
        for c in report.e1_id_mismatch_cases[:10]:
            ord_flag  = "" if c["order_match"] else "  ✗order"
            item_flag = "" if c["item_match"] else "  ✗item"
            print(f"    [{c['id']}] order: expected={c['expected_order']!r} got={c['extracted_order']!r}"
                  f"{ord_flag} | item: expected={c['expected_item']!r} got={c['extracted_item']!r}{item_flag}")

    if failed_e3:
        print(f"\n  E3 failures ({len(failed_e3)}):")
        for r in failed_e3[:10]:
            print(f"    [{r['id']}] expected={r['expected_status']!r}  "
                  f"got={r['actual_status']!r}")

    # text2sql이 행은 줬지만 값이 골든과 어긋난 케이스 (silent error — LLM SQL이 틀린 신호)
    if report.e2_mismatch_cases:
        print(f"\n  E2 value mismatches ({len(report.e2_mismatch_cases)})  "
              f"[text2sql returned a row but values differ from golden]:")
        for c in report.e2_mismatch_cases[:10]:
            print(f"    [{c['id']}] order={c['order_id']} item={c['item_no']}")
            for m in c["mismatches"]:
                print(f"      {m['column']}: expected={m['expected']!r}  actual={m['actual']!r}")

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
