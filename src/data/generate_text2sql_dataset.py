"""
src/data/generate_text2sql_dataset.py
────────────────────────────────────────────────────────────────────────────
Text-to-SQL 평가용 테스트 케이스 생성기

흐름:
  1. data/sap_erp.db (SQLite)에서 다양한 시나리오 케이스를 직접 샘플링
     - WBSTA = A / B / C 케이스
     - MARD JOIN 가능한 케이스 (available_stock 있음)
     - NULL 케이스 (no-join row)
     - 존재하지 않는 order_id (negative case)
  2. 샘플링한 실제 DB 값으로 expected_values 자동 계산
  3. 기존 text2sql_test_cases.json 포맷으로 저장

실행:
  python -m src.data.generate_text2sql_dataset
  python -m src.data.generate_text2sql_dataset --n-samples 30 --output data/eval/text2sql_test_cases_gen.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sqlite3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv()

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT_FOR_LOG = Path(__file__).resolve().parent.parent.parent
_LOG_DIR = _ROOT_FOR_LOG / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(_LOG_DIR / "data_generation.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB     = str(_ROOT / "data" / "sap_erp.db")
DEFAULT_OUTPUT = str(_ROOT / "data" / "eval" / "text2sql_test_cases_gen.json")

REQUIRED_COLUMNS = ["material_name", "quantity", "delivery_status", "delivery_date", "available_stock"]

# negative case에 사용할 가짜 order_id
NEGATIVE_ORDER_ID = "9999999999"

# ---------------------------------------------------------------------------
# 데이터 구조
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    id:               str
    description:      str
    order_id:         str
    item_no:          int
    expected_columns: list[str]
    expected_values:  dict[str, Any] | None
    golden_sql:       str          # 정답(레퍼런스) SQL — 프로덕션 도구의 _hardcoded_query 결과
    notes:            str


# ---------------------------------------------------------------------------
# DB 샘플링 쿼리 모음
# ---------------------------------------------------------------------------

# 각 시나리오를 (description_template, notes, sql) 튜플로 정의
# SQL은 (VBELN, POSNR, quantity, delivery_status, delivery_date, available_stock) 반환
_SAMPLING_QUERIES: list[tuple[str, str, str]] = [

    # ── 케이스 1: WBSTA = A (미처리) ─────────────────────────────────────────
    (
        "Order with WBSTA=A (goods movement not yet processed)",
        "WBSTA=A means goods movement not yet started",
        """
        SELECT
            vbap.VBELN          AS order_id,
            vbap.POSNR          AS item_no,
            makt.MAKTX          AS material_name,
            vbap.KWMENG         AS quantity,
            vbup.WBSTA          AS delivery_status,
            vbep.EDATU          AS delivery_date,
            NULL                AS available_stock
        FROM VBAP vbap
        LEFT JOIN MAKT makt ON vbap.MATNR = makt.MATNR AND makt.SPRAS = 'E'
        LEFT JOIN VBUP vbup ON CAST(vbap.VBELN AS INTEGER) = CAST(vbup.VBELN AS INTEGER) AND CAST(vbap.POSNR AS INTEGER) = CAST(vbup.POSNR AS INTEGER)
        LEFT JOIN VBEP vbep ON CAST(vbap.VBELN AS INTEGER) = CAST(vbep.VBELN AS INTEGER) AND CAST(vbap.POSNR AS INTEGER) = CAST(vbep.POSNR AS INTEGER)
        WHERE vbup.WBSTA = 'A'
        ORDER BY RANDOM()
        LIMIT {limit}
        """,
    ),

    # ── 케이스 2: WBSTA = B (부분 처리) ───────────────────────────────────────
    (
        "Order with WBSTA=B (partially processed)",
        "WBSTA=B means goods movement partially processed",
        """
        SELECT
            vbap.VBELN          AS order_id,
            vbap.POSNR          AS item_no,
            makt.MAKTX          AS material_name,
            vbap.KWMENG         AS quantity,
            vbup.WBSTA          AS delivery_status,
            vbep.EDATU          AS delivery_date,
            NULL                AS available_stock
        FROM VBAP vbap
        LEFT JOIN MAKT makt ON vbap.MATNR = makt.MATNR AND makt.SPRAS = 'E'
        LEFT JOIN VBUP vbup ON CAST(vbap.VBELN AS INTEGER) = CAST(vbup.VBELN AS INTEGER) AND CAST(vbap.POSNR AS INTEGER) = CAST(vbup.POSNR AS INTEGER)
        LEFT JOIN VBEP vbep ON CAST(vbap.VBELN AS INTEGER) = CAST(vbep.VBELN AS INTEGER) AND CAST(vbap.POSNR AS INTEGER) = CAST(vbep.POSNR AS INTEGER)
        WHERE vbup.WBSTA = 'B'
        ORDER BY RANDOM()
        LIMIT {limit}
        """,
    ),

    # ── 케이스 3: WBSTA = C (완전 처리) ───────────────────────────────────────
    (
        "Order with WBSTA=C (fully processed)",
        "WBSTA=C means goods movement fully completed",
        """
        SELECT
            vbap.VBELN          AS order_id,
            vbap.POSNR          AS item_no,
            makt.MAKTX          AS material_name,
            vbap.KWMENG         AS quantity,
            vbup.WBSTA          AS delivery_status,
            vbep.EDATU          AS delivery_date,
            NULL                AS available_stock
        FROM VBAP vbap
        LEFT JOIN MAKT makt ON vbap.MATNR = makt.MATNR AND makt.SPRAS = 'E'
        LEFT JOIN VBUP vbup ON CAST(vbap.VBELN AS INTEGER) = CAST(vbup.VBELN AS INTEGER) AND CAST(vbap.POSNR AS INTEGER) = CAST(vbup.POSNR AS INTEGER)
        LEFT JOIN VBEP vbep ON CAST(vbap.VBELN AS INTEGER) = CAST(vbep.VBELN AS INTEGER) AND CAST(vbap.POSNR AS INTEGER) = CAST(vbep.POSNR AS INTEGER)
        WHERE vbup.WBSTA = 'C'
        ORDER BY RANDOM()
        LIMIT {limit}
        """,
    ),

    # ── 케이스 4: MARD JOIN 가능 (available_stock 있음) ─────────────────────
    (
        "Order with available stock via MARD JOIN",
        "MARD.LABST (unrestricted stock) should be present",
        """
        SELECT
            vbap.VBELN          AS order_id,
            vbap.POSNR          AS item_no,
            makt.MAKTX          AS material_name,
            vbap.KWMENG         AS quantity,
            vbup.WBSTA          AS delivery_status,
            vbep.EDATU          AS delivery_date,
            mard.LABST          AS available_stock
        FROM VBAP vbap
        LEFT JOIN MAKT makt ON vbap.MATNR = makt.MATNR AND makt.SPRAS = 'E'
        LEFT JOIN VBUP vbup ON CAST(vbap.VBELN AS INTEGER) = CAST(vbup.VBELN AS INTEGER) AND CAST(vbap.POSNR AS INTEGER) = CAST(vbup.POSNR AS INTEGER)
        LEFT JOIN VBEP vbep ON CAST(vbap.VBELN AS INTEGER) = CAST(vbep.VBELN AS INTEGER) AND CAST(vbap.POSNR AS INTEGER) = CAST(vbep.POSNR AS INTEGER)
        LEFT JOIN MARD mard ON vbap.MATNR = mard.MATNR
        WHERE mard.LABST IS NOT NULL AND mard.LABST > 0
        ORDER BY RANDOM()
        LIMIT {limit}
        """,
    ),

    # ── 케이스 5: delivery_date만 있고 stock 없음 ────────────────────────────
    (
        "Order with delivery date only (no stock join)",
        "MARD join yields NULL — delivery_date must be from VBEP",
        """
        SELECT
            vbap.VBELN          AS order_id,
            vbap.POSNR          AS item_no,
            makt.MAKTX          AS material_name,
            vbap.KWMENG         AS quantity,
            vbup.WBSTA          AS delivery_status,
            vbep.EDATU          AS delivery_date,
            NULL                AS available_stock
        FROM VBAP vbap
        LEFT JOIN MAKT makt ON vbap.MATNR = makt.MATNR AND makt.SPRAS = 'E'
        LEFT JOIN VBUP vbup ON CAST(vbap.VBELN AS INTEGER) = CAST(vbup.VBELN AS INTEGER) AND CAST(vbap.POSNR AS INTEGER) = CAST(vbup.POSNR AS INTEGER)
        LEFT JOIN VBEP vbep ON CAST(vbap.VBELN AS INTEGER) = CAST(vbep.VBELN AS INTEGER) AND CAST(vbap.POSNR AS INTEGER) = CAST(vbep.POSNR AS INTEGER)
        LEFT JOIN MARD mard ON vbap.MATNR = mard.MATNR
        WHERE vbep.EDATU IS NOT NULL AND mard.LABST IS NULL
        ORDER BY RANDOM()
        LIMIT {limit}
        """,
    ),

    # ── 케이스 6: 비기본 POSNR (POSNR != 10) ────────────────────────────────
    (
        "Order item with non-default POSNR (item number != 10)",
        "Tests that POSNR is handled correctly for non-standard line items",
        """
        SELECT
            vbap.VBELN          AS order_id,
            vbap.POSNR          AS item_no,
            makt.MAKTX          AS material_name,
            vbap.KWMENG         AS quantity,
            vbup.WBSTA          AS delivery_status,
            vbep.EDATU          AS delivery_date,
            mard.LABST          AS available_stock
        FROM VBAP vbap
        LEFT JOIN MAKT makt ON vbap.MATNR = makt.MATNR AND makt.SPRAS = 'E'
        LEFT JOIN VBUP vbup ON CAST(vbap.VBELN AS INTEGER) = CAST(vbup.VBELN AS INTEGER) AND CAST(vbap.POSNR AS INTEGER) = CAST(vbup.POSNR AS INTEGER)
        LEFT JOIN VBEP vbep ON CAST(vbap.VBELN AS INTEGER) = CAST(vbep.VBELN AS INTEGER) AND CAST(vbap.POSNR AS INTEGER) = CAST(vbep.POSNR AS INTEGER)
        LEFT JOIN MARD mard ON vbap.MATNR = mard.MATNR
        WHERE vbap.POSNR != 10
        ORDER BY RANDOM()
        LIMIT {limit}
        """,
    ),

    # ── 케이스 7: 일반 기본 케이스 (특수 조건 없음) ──────────────────────────
    (
        "Standard order — baseline with all joins present",
        "General case: material name, delivery status, and delivery date all populated",
        """
        SELECT
            vbap.VBELN          AS order_id,
            vbap.POSNR          AS item_no,
            makt.MAKTX          AS material_name,
            vbap.KWMENG         AS quantity,
            vbup.WBSTA          AS delivery_status,
            vbep.EDATU          AS delivery_date,
            NULL                AS available_stock
        FROM VBAP vbap
        LEFT JOIN MAKT makt ON vbap.MATNR = makt.MATNR AND makt.SPRAS = 'E'
        LEFT JOIN VBUP vbup ON CAST(vbap.VBELN AS INTEGER) = CAST(vbup.VBELN AS INTEGER) AND CAST(vbap.POSNR AS INTEGER) = CAST(vbup.POSNR AS INTEGER)
        LEFT JOIN VBEP vbep ON CAST(vbap.VBELN AS INTEGER) = CAST(vbep.VBELN AS INTEGER) AND CAST(vbap.POSNR AS INTEGER) = CAST(vbep.POSNR AS INTEGER)
        WHERE makt.MAKTX IS NOT NULL
          AND vbup.WBSTA IS NOT NULL
          AND vbep.EDATU IS NOT NULL
        ORDER BY RANDOM()
        LIMIT {limit}
        """,
    ),

    # ── 케이스 8: 납기 초과 미완료 주문 (overdue) ────────────────────────────
    (
        "Overdue order — delivery date in the past, goods movement incomplete",
        "EDATU < today and WBSTA in (A, B) — tests time-sensitive filtering",
        """
        SELECT
            vbap.VBELN          AS order_id,
            vbap.POSNR          AS item_no,
            makt.MAKTX          AS material_name,
            vbap.KWMENG         AS quantity,
            vbup.WBSTA          AS delivery_status,
            vbep.EDATU          AS delivery_date,
            NULL                AS available_stock
        FROM VBAP vbap
        LEFT JOIN MAKT makt ON vbap.MATNR = makt.MATNR AND makt.SPRAS = 'E'
        LEFT JOIN VBUP vbup ON CAST(vbap.VBELN AS INTEGER) = CAST(vbup.VBELN AS INTEGER) AND CAST(vbap.POSNR AS INTEGER) = CAST(vbup.POSNR AS INTEGER)
        LEFT JOIN VBEP vbep ON CAST(vbap.VBELN AS INTEGER) = CAST(vbep.VBELN AS INTEGER) AND CAST(vbap.POSNR AS INTEGER) = CAST(vbep.POSNR AS INTEGER)
        WHERE vbep.EDATU < strftime('%Y%m%d', 'now')
          AND vbup.WBSTA IN ('A', 'B')
          AND vbep.EDATU IS NOT NULL
        ORDER BY RANDOM()
        LIMIT {limit}
        """,
    ),

    # ── 케이스 9: 미래 납기 예정 주문 ────────────────────────────────────────
    (
        "Upcoming delivery — delivery date in the future, not yet processed",
        "EDATU >= today and WBSTA = A — tests future-date query handling",
        """
        SELECT
            vbap.VBELN          AS order_id,
            vbap.POSNR          AS item_no,
            makt.MAKTX          AS material_name,
            vbap.KWMENG         AS quantity,
            vbup.WBSTA          AS delivery_status,
            vbep.EDATU          AS delivery_date,
            NULL                AS available_stock
        FROM VBAP vbap
        LEFT JOIN MAKT makt ON vbap.MATNR = makt.MATNR AND makt.SPRAS = 'E'
        LEFT JOIN VBUP vbup ON CAST(vbap.VBELN AS INTEGER) = CAST(vbup.VBELN AS INTEGER) AND CAST(vbap.POSNR AS INTEGER) = CAST(vbup.POSNR AS INTEGER)
        LEFT JOIN VBEP vbep ON CAST(vbap.VBELN AS INTEGER) = CAST(vbep.VBELN AS INTEGER) AND CAST(vbap.POSNR AS INTEGER) = CAST(vbep.POSNR AS INTEGER)
        WHERE vbep.EDATU >= strftime('%Y%m%d', 'now')
          AND vbup.WBSTA = 'A'
        ORDER BY RANDOM()
        LIMIT {limit}
        """,
    ),

    # ── 케이스 10: 재고가 주문량 이상인 경우 (stock surplus) ──────────────────
    (
        "Order where available stock exceeds ordered quantity",
        "MARD.LABST > VBAP.KWMENG — tests stock-vs-quantity comparison logic",
        """
        SELECT
            vbap.VBELN          AS order_id,
            vbap.POSNR          AS item_no,
            makt.MAKTX          AS material_name,
            vbap.KWMENG         AS quantity,
            vbup.WBSTA          AS delivery_status,
            vbep.EDATU          AS delivery_date,
            mard.LABST          AS available_stock
        FROM VBAP vbap
        LEFT JOIN MAKT makt ON vbap.MATNR = makt.MATNR AND makt.SPRAS = 'E'
        LEFT JOIN VBUP vbup ON CAST(vbap.VBELN AS INTEGER) = CAST(vbup.VBELN AS INTEGER) AND CAST(vbap.POSNR AS INTEGER) = CAST(vbup.POSNR AS INTEGER)
        LEFT JOIN VBEP vbep ON CAST(vbap.VBELN AS INTEGER) = CAST(vbep.VBELN AS INTEGER) AND CAST(vbap.POSNR AS INTEGER) = CAST(vbep.POSNR AS INTEGER)
        LEFT JOIN MARD mard ON vbap.MATNR = mard.MATNR
        WHERE mard.LABST > vbap.KWMENG
        ORDER BY RANDOM()
        LIMIT {limit}
        """,
    ),

    # ── 케이스 11: 다중 라인 아이템 주문 ─────────────────────────────────────
    (
        "Multi-line order — sales order containing more than one item",
        "Tests item disambiguation when VBELN has multiple POSNRs",
        """
        SELECT
            vbap.VBELN          AS order_id,
            vbap.POSNR          AS item_no,
            makt.MAKTX          AS material_name,
            vbap.KWMENG         AS quantity,
            vbup.WBSTA          AS delivery_status,
            vbep.EDATU          AS delivery_date,
            NULL                AS available_stock
        FROM VBAP vbap
        LEFT JOIN MAKT makt ON vbap.MATNR = makt.MATNR AND makt.SPRAS = 'E'
        LEFT JOIN VBUP vbup ON CAST(vbap.VBELN AS INTEGER) = CAST(vbup.VBELN AS INTEGER) AND CAST(vbap.POSNR AS INTEGER) = CAST(vbup.POSNR AS INTEGER)
        LEFT JOIN VBEP vbep ON CAST(vbap.VBELN AS INTEGER) = CAST(vbep.VBELN AS INTEGER) AND CAST(vbap.POSNR AS INTEGER) = CAST(vbep.POSNR AS INTEGER)
        WHERE vbap.VBELN IN (
            SELECT VBELN FROM VBAP GROUP BY VBELN HAVING COUNT(DISTINCT POSNR) > 1
        )
        ORDER BY RANDOM()
        LIMIT {limit}
        """,
    ),

    # ── 케이스 12: 재고 없음 (LABST = 0 또는 JOIN 불가) ──────────────────────
    (
        "Order with zero or no available stock",
        "MARD.LABST = 0 or NULL — tests zero-stock edge case handling",
        """
        SELECT
            vbap.VBELN                  AS order_id,
            vbap.POSNR                  AS item_no,
            makt.MAKTX                  AS material_name,
            vbap.KWMENG                 AS quantity,
            vbup.WBSTA                  AS delivery_status,
            vbep.EDATU                  AS delivery_date,
            COALESCE(mard.LABST, 0.0)   AS available_stock
        FROM VBAP vbap
        LEFT JOIN MAKT makt ON vbap.MATNR = makt.MATNR AND makt.SPRAS = 'E'
        LEFT JOIN VBUP vbup ON CAST(vbap.VBELN AS INTEGER) = CAST(vbup.VBELN AS INTEGER) AND CAST(vbap.POSNR AS INTEGER) = CAST(vbup.POSNR AS INTEGER)
        LEFT JOIN VBEP vbep ON CAST(vbap.VBELN AS INTEGER) = CAST(vbep.VBELN AS INTEGER) AND CAST(vbap.POSNR AS INTEGER) = CAST(vbep.POSNR AS INTEGER)
        LEFT JOIN MARD mard ON vbap.MATNR = mard.MATNR
        WHERE (mard.LABST IS NULL OR mard.LABST = 0)
          AND vbap.KWMENG > 0
          AND makt.MAKTX IS NOT NULL
        ORDER BY RANDOM()
        LIMIT {limit}
        """,
    ),
]

# ---------------------------------------------------------------------------
# DB 연결 헬퍼
# ---------------------------------------------------------------------------

def _query(db_path: str, sql: str, conn: sqlite3.Connection | None = None) -> list[dict]:
    """SQL 실행 → dict 리스트. conn이 주어지면 그 연결을 재사용(닫지 않음),
    없으면 일회성 연결을 열고 닫는다(하위호환). 배치 샘플링에선 conn을 재사용해
    연결 오픈 비용 + SQLite 페이지 캐시 콜드스타트를 피한다."""
    own = conn is None
    if own:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(sql).fetchall()
        return [dict(r) for r in rows]
    finally:
        if own:
            conn.close()


def _edatu_to_iso(edatu: str | None) -> str | None:
    """YYYYMMDD → YYYY-MM-DD 변환 (이미 ISO면 그대로)"""
    if edatu and len(edatu) == 8:
        return f"{edatu[:4]}-{edatu[4:6]}-{edatu[6:]}"
    return edatu


# 샘플링·캐노니컬 쿼리가 쓰는 CAST(...) 조인 키에 대한 표현식 인덱스.
# 이게 없으면 CAST 양변 조인이 인덱스를 못 타 VBAP×VBUP 중첩루프(수십억 비교)로 폭발한다
# (시나리오 쿼리 1회 ~254초 관측). 인덱스가 있으면 ~0.2초로 떨어진다. 멱등 생성.
_PERF_INDEXES = [
    "CREATE INDEX IF NOT EXISTS ix_vbup_norm ON VBUP(CAST(VBELN AS INTEGER), CAST(POSNR AS INTEGER))",
    "CREATE INDEX IF NOT EXISTS ix_vbep_norm ON VBEP(CAST(VBELN AS INTEGER), CAST(POSNR AS INTEGER))",
    "CREATE INDEX IF NOT EXISTS ix_mard_mw   ON MARD(MATNR, WERKS)",
    "CREATE INDEX IF NOT EXISTS ix_makt_m    ON MAKT(MATNR)",
]


def _ensure_perf_indexes(conn: sqlite3.Connection) -> None:
    """샘플링 성능용 인덱스를 멱등 생성. 실패해도(예: 읽기전용/락) 흐름을 끊지 않는다."""
    try:
        for ddl in _PERF_INDEXES:
            conn.execute(ddl)
        conn.commit()
    except sqlite3.Error as e:
        logger.warning("Perf index creation skipped: %s", e)


def _canonical_query_and_row(
    db_path: str, order_id: str, item_no: str | int,
    conn: sqlite3.Connection | None = None,
) -> tuple[str, dict | None]:
    """expected_values + golden_sql의 단일 출처(source of truth).

    프로덕션 프롬프트가 LLM에게 지시하는 의미와 1:1로 일치하는 레퍼런스 쿼리
    (src.tools.text2sql._hardcoded_query)를 그대로 실행해 정답 행을 만든다.
    시나리오 SQL은 (order_id, item_no) 다양성 선택에만 쓰고, 값·정답SQL은 여기서 산출한다.
    이렇게 해야 골든(값+SQL)과 프롬프트가 같은 결정적 정의를 공유한다.

    conn이 주어지면 재사용한다(배치 샘플링 성능).

    Returns: (golden_sql, 정답 행 or None)
    """
    from src.tools.text2sql import _hardcoded_query
    sql = _hardcoded_query(str(order_id), str(item_no))
    rows = _query(db_path, sql, conn=conn)
    return sql, (rows[0] if rows else None)


# ---------------------------------------------------------------------------
# 테스트 케이스 빌더
# ---------------------------------------------------------------------------

def _build_expected_values(row: dict) -> dict | None:
    """DB 행에서 expected_values dict를 구성합니다."""
    ev: dict = {}
    if row.get("material_name"):
        ev["material_name"] = row["material_name"]
    if row.get("quantity") is not None:
        ev["quantity"] = float(row["quantity"])
    if row.get("delivery_status"):
        ev["delivery_status"] = row["delivery_status"]
    if row.get("delivery_date"):
        ev["delivery_date"] = _edatu_to_iso(row["delivery_date"])
    if row.get("available_stock") is not None:
        ev["available_stock"] = float(row["available_stock"])
    return ev if ev else None


def _sample_cases(
    db_path: str,
    n_per_scenario: int,
    seed: int = 42,
) -> list[TestCase]:
    """각 샘플링 시나리오에서 n_per_scenario개씩 케이스를 수집합니다."""
    random.seed(seed)
    cases: list[TestCase] = []

    # 연결 1개 재사용(264+ 오픈 방지 + 페이지 캐시 유지) + CAST 조인 인덱스 멱등 생성.
    # 인덱스가 없으면 시나리오 쿼리의 CAST 양변 조인이 중첩루프로 폭발한다(1회 ~254초).
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    _ensure_perf_indexes(conn)
    try:
        for desc_tmpl, notes, sql_tmpl in _SAMPLING_QUERIES:
            # 넉넉하게 쿼리 후 랜덤 선택
            sql = sql_tmpl.format(limit=n_per_scenario * 5)
            try:
                rows = _query(db_path, sql, conn=conn)
            except Exception as e:
                logger.warning("Sampling failed (%s): %s", desc_tmpl[:40], e)
                continue

            sampled = random.sample(rows, min(n_per_scenario, len(rows)))
            for row in sampled:
                order_id = str(row.get("order_id", "")).strip()
                item_no  = int(row.get("item_no", 10))

                # expected_values·golden_sql은 시나리오 행이 아니라 레퍼런스(캐노니컬) 쿼리에서 산출
                golden_sql, canon = _canonical_query_and_row(db_path, order_id, item_no, conn=conn)
                cases.append(TestCase(
                    id="",  # 나중에 부여
                    description=desc_tmpl,
                    order_id=order_id,
                    item_no=item_no,
                    expected_columns=REQUIRED_COLUMNS,
                    expected_values=_build_expected_values(canon) if canon else None,
                    golden_sql=golden_sql,
                    notes=notes,
                ))
    finally:
        conn.close()

    return cases


def _add_negative_case(cases: list[TestCase]) -> None:
    """존재하지 않는 주문번호 케이스(expected_values=None)를 추가합니다.
    정답 SQL은 유효하지만 결과 행이 없는 케이스를 검증한다."""
    from src.tools.text2sql import _hardcoded_query
    cases.append(TestCase(
        id="",
        description="Non-existent order — should return no row (None result)",
        order_id=NEGATIVE_ORDER_ID,
        item_no=10,
        expected_columns=REQUIRED_COLUMNS,
        expected_values=None,
        golden_sql=_hardcoded_query(NEGATIVE_ORDER_ID, "10"),
        notes="SQL must execute without error even if no rows returned",
    ))


# ---------------------------------------------------------------------------
# 메인 생성 함수
# ---------------------------------------------------------------------------

def generate_text2sql_dataset(
    n_samples: int = 20,
    output_path: str = DEFAULT_OUTPUT,
    db_path: str = DEFAULT_DB,
    seed: int = 42,
    append: bool = False,
) -> list[dict]:
    """
    SQLite DB를 샘플링하여 text2sql 테스트 케이스를 생성합니다.

    Parameters
    ----------
    n_samples   : 시나리오당 샘플 수 (실제 총 케이스 수 ≈ n_samples × 시나리오 수)
    output_path : 출력 JSON 경로
    db_path     : sap_erp.db 경로
    seed        : 랜덤 시드 (재현성)
    append      : 기존 파일에 추가 여부
    """
    if not Path(db_path).exists():
        raise FileNotFoundError(
            f"DB file not found: {db_path}\n"
            "Check the paths.erp_db configuration in configs.yaml."
        )

    logger.info("Starting DB sampling: %s", db_path)
    cases = _sample_cases(db_path, n_samples, seed)
    _add_negative_case(cases)

    # 중복 (order_id, item_no) 제거
    seen: set[tuple] = set()
    deduped: list[TestCase] = []
    for c in cases:
        key = (c.order_id, c.item_no)
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    cases = deduped

    # id 부여
    start_idx = 1
    if append:
        existing_path = Path(output_path)
        if existing_path.exists():
            try:
                existing = json.loads(existing_path.read_text(encoding="utf-8"))
                start_idx = len(existing) + 1
            except Exception:
                pass

    for idx, c in enumerate(cases, start=start_idx):
        c.id = f"tc_{idx:03d}"

    # dict 변환
    records = [asdict(c) for c in cases]

    # append 처리
    if append:
        existing_path = Path(output_path)
        if existing_path.exists():
            try:
                existing = json.loads(existing_path.read_text(encoding="utf-8"))
                records = existing + records
            except Exception as e:
                logger.warning("Failed to read existing file: %s", e)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    # Summary
    print("\n" + "=" * 60)
    print("  Text-to-SQL Dataset Generation Results")
    print("=" * 60)
    print(f"  DB Path         : {db_path}")
    print(f"  Scenarios       : {len(_SAMPLING_QUERIES) + 1}")
    print(f"  Generated Cases : {len(cases)}")
    print(f"  Output File     : {output_path}")
    print("=" * 60)

    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text-to-SQL 평가용 테스트 케이스 생성")
    parser.add_argument(
        "--n-samples", type=int, default=10,
        help="시나리오당 샘플 수 (기본: 10, 총 케이스 ≈ 10 × 시나리오 수)",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"출력 JSON 경로 (기본: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--db", default=DEFAULT_DB,
        help=f"SQLite DB 경로 (기본: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="랜덤 시드 (기본: 42)",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="기존 출력 파일에 결과를 추가",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_text2sql_dataset(
        n_samples=args.n_samples,
        output_path=args.output,
        db_path=args.db,
        seed=args.seed,
        append=args.append,
    )
