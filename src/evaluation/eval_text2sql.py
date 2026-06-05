"""
src/evaluation/eval_text2sql.py
Text-to-SQL 성공률 평가

판정 기준:
  [P1] SQL 구문 실행 가능 여부          (syntax / runtime error → FAIL)
  [P2] 응답 시간 10초 이내              (timeout → FAIL)
  [P3] 필수 alias 컬럼 4개 포함 여부    (quantity, delivery_status, delivery_date, available_stock)
  [P4] expected_values 일치 여부        (expected_values=None은 SQL 정상 실행이면 PASS;
                                         값이 명시됐는데 row가 없으면 FAIL — 이전엔 skip되어 거짓 PASS였음)
  [P5] 생성 기준(criteria) 준수 여부    (LLM이 시스템 프롬프트의 핵심 규칙을 지켰는지 정적 검사;
                                         실행 결과가 아닌 "어떤 기준으로 SQL을 만들었는지" 확인용)

목표: P1+P2+P3+P4 기준 95% 이상 (P4는 expected_values=None이면 자동 PASS) / P5(criteria) 별도 집계
      → 이전엔 P1+P2+P3만 봤더니 거짓 100%가 나왔음. P4가 실제 정확도를 담보하는 핵심.

실행:
  python -m src.evaluation.eval_text2sql
  python -m src.evaluation.eval_text2sql --report reports/worker_a_eval_result.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.tools.text2sql import build_validation_query, _hardcoded_query

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
    passed:      bool           # P1+P2+P3+P4 모두 통과 (P4=None이면 P1+P2+P3만으로 판정)
    p1_syntax:   bool = False   # SQL 실행 가능
    p2_timeout:  bool = False   # 10초 이내
    p3_columns:  bool = False   # 필수 alias 존재
    p4_values:   bool | None = None  # 기대값 일치 (None=검증 생략)
    p5_criteria: bool = False   # 생성 기준 모두 충족 (score == total)
    elapsed_ms:  float = 0.0
    generated_sql: str = ""
    actual_result: Any = None
    error: str = ""
    sql_strategy:  str = ""     # "primary" | "fallback" | "hardcoded"
    p4_mismatches: list = field(default_factory=list)  # [{column, expected, actual}, ...]
    criteria_checks: dict = field(default_factory=dict)  # {rule_name: bool, ...}
    criteria_score: int = 0     # 통과한 rule 수
    criteria_total: int = 0     # 검사한 rule 수
    # ── Fallback 추적 (운영 로깅용 — eval에서도 같은 시그널을 본다) ─────────────
    fallback_triggered:  bool = False  # 운영에서 hardcoded fallback이 (혹은 트리거됐을) 케이스
    fallback_reason:     str  = ""     # "llm_fail" | "primary_zero_rows" | ""
    hardcoded_recovered: bool | None = None  # primary_zero_rows일 때 hardcoded가 row를 잡았는지


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
    p5_pass:       int = 0   # criteria 전 규칙 통과
    overall_pass:  int = 0   # P1+P2+P3+P4 모두 통과 (P4=None은 PASS로 간주)
    success_rate:  float = 0.0  # overall_pass / total
    p4_accuracy:   float = 0.0  # p4_pass / p4_total
    p5_compliance: float = 0.0  # p5_pass / total (전 규칙 만족 비율)
    criteria_rule_pass: dict = field(default_factory=dict)  # {rule_name: 통과 케이스 수}
    criteria_avg_score: float = 0.0  # 케이스당 평균 점수 (score/total)
    strategy_primary:    int = 0  # primary LLM으로 생성된 케이스 수 (build 시점)
    strategy_hardcoded:  int = 0  # build 단계에서 hardcoded로 폴백된 수 (LLM 호출/파싱 실패)
    fallback_zero_rows:  int = 0  # LLM SQL이 실행은 됐지만 0건 반환 → 운영에서는 hardcoded로 재시도되는 케이스
    fallback_zero_rows_recoverable: int = 0  # 그 중 hardcoded로 재시도하면 row가 잡혀 LLM 잘못이 입증되는 수
    p4_mismatch_cases: list[dict] = field(default_factory=list)  # P4 불일치 케이스 상세
    fallback_cases:    list[dict] = field(default_factory=list)  # fallback이 일어났거나 트리거됐을 케이스 상세
    results:       list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# P5: SQL 생성 기준(criteria) 정적 검사
# ---------------------------------------------------------------------------
#
# SQL 문법/실행은 P1이 보고, 여기서는 "LLM이 시스템 프롬프트의 핵심 규칙을 따랐는가"를
# 정적 정규식으로 본다. row가 비어도 — 즉 결과적으로 0건이 나와도 — 규칙을 어디서
# 어겼는지 보고하기 위함이다.
#
# 규칙 (text2sql.py _SYSTEM_PROMPT 참조):
#   r1 cast_vbeln          : VBELN을 CAST(... AS INTEGER)로 정규화        (rule 4)
#   r2 cast_posnr          : POSNR을 CAST(... AS INTEGER)로 정규화        (rule 4)
#   r3 mard_sum_aggregation: 저장위치별 다행 → SUM(LABST) 집계             (rule 3a)
#   r4 makt_lang_handling  : 언어별 다행 → SPRAS 기준 우선순위              (rule 3b)
#   r5 min_edatu           : 일정라인 가장 이른 날짜 → MIN(EDATU)           (rule 3c)
#   r6 coalesce_zero       : NULL stock → 0 으로 보정 COALESCE(..., 0)     (rule 3a)

_CRITERIA_PATTERNS = {
    "cast_vbeln":           r"CAST\s*\(\s*[\w.]*VBELN\s+AS\s+INTEGER\s*\)",
    "cast_posnr":           r"CAST\s*\(\s*[\w.]*POSNR\s+AS\s+INTEGER\s*\)",
    "mard_sum_aggregation": r"SUM\s*\(\s*[\w.]*LABST\s*\)",
    "makt_lang_handling":   r"\bSPRAS\b",
    "min_edatu":            r"MIN\s*\(\s*[\w.]*EDATU\s*\)",
    "coalesce_zero":        r"COALESCE\s*\([^()]*,\s*0\s*\)",
}


def _check_criteria(sql: str) -> tuple[dict, int, int]:
    """LLM이 시스템 프롬프트의 어떤 규칙을 지켰는지 정적 검사.

    Returns:
        (checks, score, total) — checks는 {rule_name: bool}, score/total은 통과/전체 규칙 수
    """
    s = sql.upper()
    checks = {name: bool(re.search(pat, s, re.IGNORECASE)) for name, pat in _CRITERIA_PATTERNS.items()}
    score = sum(checks.values())
    total = len(checks)
    return checks, score, total

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

    # ── P5: 생성 기준(criteria) 정적 검사 ─────────────────────────────────────
    # SQL이 실행되든 안 되든, 어떤 규칙을 지켰는지 항상 기록한다.
    checks, score, total = _check_criteria(sql)
    cr.criteria_checks = checks
    cr.criteria_score  = score
    cr.criteria_total  = total
    cr.p5_criteria     = (score == total)

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

    # ── Fallback 추적 (운영에서 hardcoded 폴백이 실제 일어났거나, 일어났을 케이스) ─────
    # eval은 build_validation_query만 호출하므로 운영 코드(run_validation_query)의
    # "primary 0건 → hardcoded 재시도" 경로는 자동으로 타지 않는다.
    # 같은 시그널을 재현하기 위해 여기서 동일 조건을 직접 확인하고 hardcoded 결과까지 비교.
    if strategy == "hardcoded_llm_fail":
        cr.fallback_triggered = True
        cr.fallback_reason    = "llm_fail"
        logger.warning(
            "[%s] FALLBACK(llm_fail) | LLM 호출 실패 → hardcoded 쿼리로 평가됨",
            tc["id"],
        )
    elif strategy == "primary" and cr.actual_result is None:
        cr.fallback_triggered = True
        cr.fallback_reason    = "primary_zero_rows"
        # hardcoded 쿼리로 재실행해서 LLM 잘못인지 / 실데이터 없음인지 가른다
        try:
            hc_rows, _ = _execute_sql(_hardcoded_query(order_id, str(item_no)), db_path)
            cr.hardcoded_recovered = len(hc_rows) > 0
            logger.warning(
                "[%s] FALLBACK(primary_zero_rows) | LLM SQL 0건, hardcoded 재시도 → %s",
                tc["id"],
                "row 발견 (LLM 쿼리 오류 강한 신호)" if cr.hardcoded_recovered
                    else "여전히 0건 (실데이터 부재 가능)",
            )
        except Exception as e:
            logger.warning("[%s] hardcoded 재시도 실패: %s", tc["id"], e)
            cr.hardcoded_recovered = None
    # ── P4: 기대값 일치 ───────────────────────────────────────────────────────
    if expected is not None:
        if cr.actual_result is None:
            # 이전 버그: 여기서 p4_values=None으로 두면 평가 집계에서 통째로 빠져
            # "거짓 100% PASS"가 나왔다. 기대값이 있는데 row가 0건이면 명백히 FAIL.
            cr.p4_values = False
            cr.p4_mismatches = [{
                "column":  "<row>",
                "expected": "row exists matching expected_values",
                "actual":   "no row returned (SQL ran but matched 0 rows)",
            }]
            cr.error = (cr.error + " | " if cr.error else "") + \
                       "P4 FAIL: expected_values set but SQL returned 0 rows"
        else:
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
        # expected_values=null -> row가 없어야 정상 OR 컬럼이 맞으면 OK
        cr.p4_values = True  # no-row 케이스는 SQL만 정상이면 PASS

    # ── 최종 통과 판정 ────────────────────────────────────────────────────────
    # P4가 None(=검증 대상 아님)이면 기술 통과(P1+P2+P3)만으로 PASS,
    # P4가 True/False면 그것까지 포함해 PASS 결정. False면 즉시 FAIL.
    p4_ok = cr.p4_values is not False
    cr.passed = cr.p1_syntax and cr.p2_timeout and cr.p3_columns and p4_ok

    status = "PASS" if cr.passed else "FAIL"
    logger.info(
        "[%s] %s | elapsed=%.0fms | p1=%s p2=%s p3=%s p4=%s p5=%d/%d strategy=%s",
        tc["id"], status, cr.elapsed_ms,
        cr.p1_syntax, cr.p2_timeout, cr.p3_columns, cr.p4_values,
        cr.criteria_score, cr.criteria_total, cr.sql_strategy,
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

    # P5 rule별 누적 카운터 — 어떤 규칙이 자주 누락되는지 보기 위함
    rule_pass_counter: dict[str, int] = {name: 0 for name in _CRITERIA_PATTERNS.keys()}
    criteria_score_sum = 0

    for tc in test_cases:
        cr = _evaluate_case(tc, db_path)
        report.results.append(asdict(cr))

        if cr.p1_syntax:   report.p1_pass += 1
        if cr.p2_timeout:  report.p2_pass += 1
        if cr.p3_columns:  report.p3_pass += 1
        if cr.p5_criteria: report.p5_pass += 1
        if cr.passed:      report.overall_pass += 1

        if cr.sql_strategy == "primary":              report.strategy_primary += 1
        elif cr.sql_strategy == "hardcoded_llm_fail": report.strategy_hardcoded += 1

        if cr.fallback_triggered:
            if cr.fallback_reason == "primary_zero_rows":
                report.fallback_zero_rows += 1
                if cr.hardcoded_recovered:
                    report.fallback_zero_rows_recoverable += 1
            report.fallback_cases.append({
                "id":           cr.id,
                "description":  cr.description,
                "order_id":     cr.order_id,
                "item_no":      cr.item_no,
                "reason":       cr.fallback_reason,
                "hardcoded_recovered": cr.hardcoded_recovered,
                "sql_strategy": cr.sql_strategy,
            })

        # P5: rule별 통과 카운트
        for rule_name, ok in cr.criteria_checks.items():
            if ok:
                rule_pass_counter[rule_name] = rule_pass_counter.get(rule_name, 0) + 1
        criteria_score_sum += cr.criteria_score

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
    report.p5_compliance = round(report.p5_pass / n, 4) if n else 0.0
    report.criteria_rule_pass = rule_pass_counter
    total_rules = len(_CRITERIA_PATTERNS)
    report.criteria_avg_score = round(criteria_score_sum / (n * total_rules), 4) if n else 0.0

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
    print(f"  P5 Criteria    : {report.p5_pass}/{n}  ({report.p5_compliance*100:.1f}%)  "
          f"[avg score {report.criteria_avg_score*100:.1f}%]")
    target_met = report.success_rate >= 0.95
    print(f"  Target Met     : {'YES' if target_met else 'NO'}")
    print("-" * 60)
    print(f"  SQL Strategy   : primary={report.strategy_primary}  "
          f"hardcoded(llm_fail)={report.strategy_hardcoded}")
    print(f"  Fallback (운영 시 hardcoded 재시도가 트리거됐을 케이스):")
    print(f"    primary_zero_rows         : {report.fallback_zero_rows}/{n}  "
          f"(LLM SQL이 0건 → hardcoded 재시도 발생)")
    print(f"    └─ hardcoded로 복구 가능 : {report.fallback_zero_rows_recoverable}/{report.fallback_zero_rows or 1}  "
          f"(LLM 쿼리가 실제로 잘못됐다는 신호)")
    print("-" * 60)
    print("  P5 Rule-level pass rate (어떤 기준으로 SQL을 만들었는지):")
    for rule_name, cnt in report.criteria_rule_pass.items():
        pct = (cnt / n * 100) if n else 0.0
        print(f"    {rule_name:<22}: {cnt}/{n}  ({pct:.1f}%)")
    print("=" * 60)

    # 실패 케이스 상세 출력
    failed = [r for r in report.results if not r["passed"]]
    if failed:
        print(f"\n  Failed cases ({len(failed)}):")
        for r in failed:
            print(f"    [{r['id']}] {r['description']}")
            print(f"      Error: {r['error']}")

    # Fallback 케이스 상세 출력 (LLM이 어디서 실패해 hardcoded로 떨어졌는지)
    if report.fallback_cases:
        print(f"\n  Fallback cases ({len(report.fallback_cases)}):")
        for c in report.fallback_cases:
            recov = c.get("hardcoded_recovered")
            recov_str = (
                "hardcoded로 row 회수 (LLM SQL 오류)" if recov is True
                else "hardcoded도 0건"                if recov is False
                else "n/a"
            )
            print(f"    [{c['id']}] reason={c['reason']}  strategy={c['sql_strategy']}  → {recov_str}")
            print(f"      order_id={c['order_id']} item_no={c['item_no']} — {c['description']}")

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
