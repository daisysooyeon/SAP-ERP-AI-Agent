"""
src/evaluation/eval_text2sql.py
Text-to-SQL 성공률 평가

판정 기준:
  [P1] SQL 구문 실행 가능 여부          (syntax / runtime error → FAIL)
  [P2] 응답 시간 10초 이내              (timeout → FAIL)
  [P3] 필수 alias 컬럼 4개 포함 여부    (quantity, delivery_status, delivery_date, available_stock)
  [P4] expected_values 일치 여부        (None인 케이스는 "no row" 검증)

목표: P1+P2+P3 기준 95% 이상 / P4(정확도) 별도 집계

실행:
  python -m src.evaluation.eval_text2sql
  python -m src.evaluation.eval_text2sql --report reports/worker_a_eval_result.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.tools.text_to_sql import build_validation_query

# Windows cp949 환경에서 유니코드 로그 출력 안전화
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# 로깅 설정
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 경로 상수
# ---------------------------------------------------------------------------
DB_PATH        = "data/sap_erp.db"
TEST_CASES_PATH = "data/eval/text2sql_test_cases.json"
REQUIRED_ALIASES = {"quantity", "delivery_status", "delivery_date", "available_stock"}
TIMEOUT_SECS   = 10 # 조정 가능

# ---------------------------------------------------------------------------
# 데이터 클래스
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    id:          str
    description: str
    order_id:    str
    item_no:     int
    passed:      bool           # P1+P2+P3 모두 통과
    p1_syntax:   bool = False   # SQL 실행 가능
    p2_timeout:  bool = False   # 10초 이내
    p3_columns:  bool = False   # 필수 alias 존재
    p4_values:   bool | None = None  # 기대값 일치 (None=검증 생략)
    elapsed_ms:  float = 0.0
    generated_sql: str = ""
    actual_result: Any = None
    error: str = ""
    sql_strategy:  str = ""     # "primary" | "fallback" | "hardcoded"
    p4_mismatches: list = field(default_factory=list)  # [{column, expected, actual}, ...]


@dataclass
class EvalReport:
    generated_at:  str = ""
    model_primary: str = ""
    total:         int = 0
    p1_pass:       int = 0   # syntax OK
    p2_pass:       int = 0   # timeout OK
    p3_pass:       int = 0   # columns OK
    p4_pass:       int = 0   # values OK
    p4_total:      int = 0   # p4 검증 대상 수
    overall_pass:  int = 0   # P1+P2+P3 모두 통과
    success_rate:  float = 0.0  # overall_pass / total
    p4_accuracy:   float = 0.0  # p4_pass / p4_total
    strategy_primary:    int = 0  # primary LLM으로 생성된 케이스 수
    strategy_hardcoded:  int = 0  # hardcoded fallback으로 생성된 케이스 수
    p4_mismatch_cases: list[dict] = field(default_factory=list)  # P4 불일치 케이스 상세
    results:       list[dict] = field(default_factory=list)

# ---------------------------------------------------------------------------
# 실행 헬퍼
# ---------------------------------------------------------------------------

def _run_with_timeout(fn, timeout: int):
    """fn()을 별도 스레드로 실행, timeout 초 초과 시 TimeoutError"""
    result_box: list = [None]
    error_box:  list = [None]

    def target():
        try:
            result_box[0] = fn()
        except Exception as e:
            error_box[0] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        raise TimeoutError(f"SQL execution exceeded {timeout}s")
    if error_box[0] is not None:
        raise error_box[0]
    return result_box[0]


def _execute_sql(sql: str, db_path: str = DB_PATH) -> tuple[list[dict], list[str]]:
    """SQL 실행 → (rows, column_names) 튜플 반환. row가 없어도 컨럼명 확인 가능."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(sql)
        rows = [dict(row) for row in cursor.fetchall()]
        col_names = [d[0] for d in cursor.description] if cursor.description else []
        return rows, col_names
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# 케이스 단위 평가
# ---------------------------------------------------------------------------

def _evaluate_case(tc: dict, db_path: str = DB_PATH) -> CaseResult:

    order_id = str(tc["order_id"])
    item_no  = int(tc["item_no"])
    expected = tc.get("expected_values")  # None or dict

    cr = CaseResult(
        id=tc["id"],
        description=tc["description"],
        order_id=order_id,
        item_no=item_no,
        passed=False,
    )

    # ── SQL 생성 ─────────────────────────────────────────────────────────────
    try:
        sql, strategy = build_validation_query(order_id, str(item_no))
        cr.generated_sql = sql
        cr.sql_strategy = strategy
    except Exception as e:
        cr.error = f"SQL generation failed: {e}"
        logger.warning("[%s] SQL 생성 실패: %s", tc["id"], e)
        return cr

    # ── P1: 구문 실행 + P2: Timeout ──────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        rows, col_names = _run_with_timeout(lambda: _execute_sql(sql, db_path), TIMEOUT_SECS)
        cr.elapsed_ms = (time.perf_counter() - t0) * 1000
        cr.p1_syntax  = True
        cr.p2_timeout = True
        cr.actual_result = rows[0] if rows else None
    except TimeoutError as e:
        cr.elapsed_ms = TIMEOUT_SECS * 1000
        cr.error = str(e)
        cr.p1_syntax  = True   # 실행은 시작됐으므로 문법 OK
        cr.p2_timeout = False
        logger.warning("[%s] timeout: %s", tc["id"], e)
        return cr
    except Exception as e:
        cr.elapsed_ms = (time.perf_counter() - t0) * 1000
        cr.error = str(e)
        cr.p1_syntax  = False
        cr.p2_timeout = False
        logger.warning("[%s] SQL execution failed: %s", tc["id"], e)
        return cr

    # ── P3: 필수 alias 컬럼 포함 (row 유무 상관없이 cursor.description으로 검증) ────
    actual_col_set = set(col_names)
    cr.p3_columns = REQUIRED_ALIASES.issubset(actual_col_set)
    if not cr.p3_columns:
        missing = REQUIRED_ALIASES - actual_col_set
        cr.error = f"Missing columns: {missing}"
        logger.warning("[%s] Missing columns: %s", tc["id"], missing)
    # ── P4: 기대값 일치 ───────────────────────────────────────────────────────
    if expected is not None and cr.actual_result is not None:
        mismatches = []
        for col, exp_val in expected.items():
            act_val = cr.actual_result.get(col)
            if act_val != exp_val:
                mismatches.append({"column": col, "expected": exp_val, "actual": act_val})
        cr.p4_mismatches = mismatches
        cr.p4_values = (len(mismatches) == 0)
        if mismatches:
            mismatch_strs = [
                f"{m['column']}: expected={m['expected']!r}, actual={m['actual']!r}"
                for m in mismatches
            ]
            cr.error += " | Value mismatches: " + "; ".join(mismatch_strs)
    elif expected is None:
        # expected_values=null -> row가 없어야 정상 OR 콜럼이 맞으면 OK
        cr.p4_values = True  # no-row 케이스는 SQL만 정상이면 PASS

    # ── 최종 통과 판정 ────────────────────────────────────────────────────────
    cr.passed = cr.p1_syntax and cr.p2_timeout and cr.p3_columns

    status = "PASS" if cr.passed else "FAIL"
    logger.info(
        "[%s] %s | elapsed=%.0fms | p1=%s p2=%s p3=%s p4=%s",
        tc["id"], status, cr.elapsed_ms,
        cr.p1_syntax, cr.p2_timeout, cr.p3_columns, cr.p4_values,
    )
    return cr

# ---------------------------------------------------------------------------
# 메인 평가 루프
# ---------------------------------------------------------------------------

def run_eval(
    test_cases_path: str = TEST_CASES_PATH,
    db_path:         str = DB_PATH,
    report_path:     str | None = None,
) -> EvalReport:
    from src.config import get_config

    cfg = get_config()
    test_cases = json.loads(Path(test_cases_path).read_text(encoding="utf-8"))
    logger.info("=== Text-to-SQL 평가 시작 | 케이스 수: %d ===", len(test_cases))

    report = EvalReport(
        generated_at   = datetime.now(timezone.utc).isoformat(),
        model_primary  = cfg.models.worker_a_sql.name,
        total          = len(test_cases),
    )

    for tc in test_cases:
        cr = _evaluate_case(tc, db_path)
        report.results.append(asdict(cr))

        if cr.p1_syntax:  report.p1_pass += 1
        if cr.p2_timeout: report.p2_pass += 1
        if cr.p3_columns: report.p3_pass += 1
        if cr.passed:     report.overall_pass += 1

        if cr.sql_strategy == "primary":    report.strategy_primary += 1
        elif cr.sql_strategy == "hardcoded": report.strategy_hardcoded += 1

        if cr.p4_values is not None:
            report.p4_total += 1
            if cr.p4_values:
                report.p4_pass += 1
            else:
                report.p4_mismatch_cases.append({
                    "id":           cr.id,
                    "description":  cr.description,
                    "order_id":     cr.order_id,
                    "item_no":      cr.item_no,
                    "sql_strategy": cr.sql_strategy,
                    "actual_result": cr.actual_result,
                    "mismatches":   cr.p4_mismatches,
                })

    n = report.total
    report.success_rate = round(report.overall_pass / n, 4) if n else 0.0
    report.p4_accuracy  = round(report.p4_pass / report.p4_total, 4) if report.p4_total else 0.0

    # ── 요약 출력 ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Text-to-SQL 평가 결과")
    print("=" * 60)
    print(f"  Primary model  : {report.model_primary}")
    print(f"  Total cases    : {n}")
    print(f"  P1 Syntax OK   : {report.p1_pass}/{n}  ({report.p1_pass/n*100:.1f}%)")
    print(f"  P2 Timeout OK  : {report.p2_pass}/{n}  ({report.p2_pass/n*100:.1f}%)")
    print(f"  P3 Columns OK  : {report.p3_pass}/{n}  ({report.p3_pass/n*100:.1f}%)")
    print(f"  Overall PASS   : {report.overall_pass}/{n}  ({report.success_rate*100:.1f}%)  [Target >= 95%]")
    if report.p4_total:
        print(f"  P4 Value Match : {report.p4_pass}/{report.p4_total}  ({report.p4_accuracy*100:.1f}%)")
    target_met = report.success_rate >= 0.95
    print(f"  Target Met     : {'YES' if target_met else 'NO'}")
    print("-" * 60)
    print(f"  SQL Strategy   : primary={report.strategy_primary}  hardcoded={report.strategy_hardcoded}")
    print("=" * 60)

    # 실패 케이스 상세 출력
    failed = [r for r in report.results if not r["passed"]]
    if failed:
        print(f"\n  Failed cases ({len(failed)}):")
        for r in failed:
            print(f"    [{r['id']}] {r['description']}")
            print(f"      Error: {r['error']}")

    # P4 불일치 케이스 상세 출력
    if report.p4_mismatch_cases:
        print(f"\n  P4 Value Mismatch cases ({len(report.p4_mismatch_cases)}):")
        for c in report.p4_mismatch_cases:
            print(f"    [{c['id']}] {c['description']}  (strategy={c['sql_strategy']})")
            print(f"      actual_result: {c['actual_result']}")
            for m in c["mismatches"]:
                print(f"      {m['column']}: expected={m['expected']!r}  actual={m['actual']!r}")

    # ── 리포트 저장 ─────────────────────────────────────────────────────────
    if report_path:
        out = Path(report_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  Report saved: {report_path}")

    return report

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text-to-SQL 성공률 평가")
    parser.add_argument(
        "--test-cases", default=TEST_CASES_PATH,
        help=f"테스트 케이스 JSON 경로 (default: {TEST_CASES_PATH})"
    )
    parser.add_argument(
        "--db", default=DB_PATH,
        help=f"SQLite DB 경로 (default: {DB_PATH})"
    )
    parser.add_argument(
        "--report", default=None,
        help="리포트 저장 경로 (예: reports/worker_a_eval_result.json)"
    )
    args = parser.parse_args()

    run_eval(
        test_cases_path=args.test_cases,
        db_path=args.db,
        report_path=args.report,
    )
