"""
src/tools/text_to_sql.py
Text-to-SQL: dd03l 스키마 컨텍스트 기반 SQLite 검증 쿼리 생성

흐름:
  1. LLM (worker_a_sql)   → SQL 생성
  2. hardcoded fallback   → LLM 실패 시
"""

import logging
import os
import re
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import get_config

logger = logging.getLogger(__name__)

from pathlib import Path as _Path
DB_PATH = str(_Path(__file__).resolve().parent.parent.parent / "data" / "sap_erp.db")

# ---------------------------------------------------------------------------
# SAP ABAP 딕셔너리 기반 스키마 컨텍스트
# ---------------------------------------------------------------------------

SCHEMA_CONTEXT = """\
Table: VBAK (Sales Order Header)
  VBELN: Sales order number (PK, TEXT 10-digit zero-padded), KUNNR: Customer code, AUDAT: Order date

Table: VBAP (Sales Order Item)
  VBELN: Sales order number (FK), POSNR: Item number (INTEGER), MATNR: Material number,
  KWMENG: Requested quantity (REAL), NETPR: Net price (REAL), ARKTX: Short text for sales order item

Table: VBEP (Delivery Schedule)
  VBELN: Sales order number, POSNR: Item number, EDATU: Delivery date (TEXT YYYYMMDD), WMENG: Scheduled delivery quantity

Table: VBUP (Item Status)
  VBELN: Sales order number, POSNR: Item number,
  WBSTA: Goods movement status (A=Not yet processed, B=Partially processed, C=Fully processed)

Table: MARD (Plant Stock)
  MATNR: Material number, WERKS: Plant, LGORT: Storage location, LABST: Unrestricted-use stock (REAL)

Table: MAKT (Material Descriptions)
  MATNR: Material number, SPRAS: Language key, MAKTX: Material description

Notes:
  - Database engine is SQLite.
  - VBELN is stored inconsistently: some rows are 10-digit zero-padded ('0000015353'),
    others are raw numbers ('6105'). NEVER match VBELN with plain '=' equality.
    Always normalize both sides numerically:
        CAST(vbap.VBELN AS INTEGER) = CAST('<order_id>' AS INTEGER)
  - POSNR is an INTEGER (e.g. 10, 20). Match with CAST(vbap.POSNR AS INTEGER) = CAST('<item_no>' AS INTEGER).
  - EDATU is stored as TEXT in YYYYMMDD format.
"""

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a Text-to-SQL expert for SAP ERP data stored in SQLite.

Given the schema below and a request, generate a single valid SQLite SELECT query.

═════════════════ HARD RULES (violating ANY of these = wrong answer) ═════════════════

1. Output ONLY the raw SQL — no markdown, no explanation, no code fences.

2. The result must be exactly ONE row with these aliases (use these EXACT alias names):
       material_name, quantity, delivery_status, delivery_date, available_stock
   Use LEFT JOINs so missing data yields NULL/0, never zero rows.

3. CRITICAL — MARD, MAKT, VBEP have MULTIPLE rows per material/item (one row per
   storage location / language / schedule line). Direct JOIN + LIMIT 1 picks an
   ARBITRARY (often NULL) row. You MUST aggregate them in subqueries:

     available_stock — SUM stock across ALL storage locations, then COALESCE to 0:
         LEFT JOIN (SELECT MATNR, SUM(LABST) AS total_stock FROM MARD GROUP BY MATNR) stock
                ON stock.MATNR = vbap.MATNR
       SELECT COALESCE(stock.total_stock, 0) AS available_stock

     material_name — prefer English (SPRAS='E'), else any language:
         LEFT JOIN (SELECT MATNR,
                      COALESCE(MAX(CASE WHEN SPRAS='E' THEN MAKTX END), MIN(MAKTX)) AS material_name
                    FROM MAKT GROUP BY MATNR) makt ON makt.MATNR = vbap.MATNR

     delivery_date — earliest schedule line (MIN(EDATU)):
         LEFT JOIN (SELECT VBELN, POSNR, MIN(EDATU) AS EDATU FROM VBEP GROUP BY VBELN, POSNR) vbep
                ON CAST(vbep.VBELN AS INTEGER) = CAST(vbap.VBELN AS INTEGER)
               AND CAST(vbep.POSNR AS INTEGER) = CAST(vbap.POSNR AS INTEGER)

4. ⚠️ MOST COMMON FAILURE — read carefully ⚠️
   VBELN is stored INCONSISTENTLY in the DB: some rows are 10-digit zero-padded
   ('0000015353'), others are raw ('6105'). Plain string equality WILL fail and
   return 0 rows for most inputs. You MUST normalize numerically on BOTH sides
   of every VBELN comparison:

       WHERE CAST(vbap.VBELN AS INTEGER) = CAST('<order_id>' AS INTEGER)
         AND CAST(vbap.POSNR AS INTEGER) = CAST('<item_no>'  AS INTEGER)

   This applies to ALL VBELN/POSNR comparisons — WHERE clauses AND every
   JOIN condition that uses VBELN or POSNR. Do NOT zero-pad the input yourself;
   CAST handles it.

5. End with LIMIT 1 (with proper aggregation above, this only guards against
   duplicate VBAP rows).

6. Use exact column and table names as shown in the schema.

═════════════════ EXAMPLES ═════════════════

❌ WRONG — this is what a careless LLM produces. It returns 0 rows because
   VBELN is stored padded ('0000000040') but the request says '40', and MARD/MAKT/VBEP
   are joined directly, picking arbitrary rows:

   SELECT MAKT.MAKTX AS material_name, VBAP.KWMENG AS quantity,
          VBUP.WBSTA AS delivery_status, VBEP.EDATU AS delivery_date,
          COALESCE(MARD.LABST, 0) AS available_stock
   FROM VBAP
   LEFT JOIN MAKT ON VBAP.MATNR = MAKT.MATNR
   LEFT JOIN VBUP ON VBAP.VBELN = VBUP.VBELN AND VBAP.POSNR = VBUP.POSNR
   LEFT JOIN VBEP ON VBAP.VBELN = VBEP.VBELN AND VBAP.POSNR = VBEP.POSNR
   LEFT JOIN MARD ON VBAP.MATNR = MARD.MATNR
   WHERE VBAP.VBELN = '40' AND VBAP.POSNR = 10
   LIMIT 1;
   -- violations: rule 3 (no aggregation), rule 4 (no CAST on VBELN/POSNR)

✅ CORRECT — aggregates MARD/MAKT/VBEP in subqueries, CASTs every VBELN/POSNR
   comparison to INTEGER, COALESCEs stock to 0:

   SELECT
       makt.material_name              AS material_name,
       vbap.KWMENG                     AS quantity,
       vbup.WBSTA                      AS delivery_status,
       vbep.EDATU                      AS delivery_date,
       COALESCE(stock.total_stock, 0)  AS available_stock
   FROM VBAP vbap
   LEFT JOIN (
       SELECT MATNR,
              COALESCE(MAX(CASE WHEN SPRAS='E' THEN MAKTX END), MIN(MAKTX)) AS material_name
       FROM MAKT GROUP BY MATNR
   ) makt ON makt.MATNR = vbap.MATNR
   LEFT JOIN VBUP vbup
          ON CAST(vbup.VBELN AS INTEGER) = CAST(vbap.VBELN AS INTEGER)
         AND CAST(vbup.POSNR AS INTEGER) = CAST(vbap.POSNR AS INTEGER)
   LEFT JOIN (
       SELECT VBELN, POSNR, MIN(EDATU) AS EDATU FROM VBEP GROUP BY VBELN, POSNR
   ) vbep ON CAST(vbep.VBELN AS INTEGER) = CAST(vbap.VBELN AS INTEGER)
        AND CAST(vbep.POSNR AS INTEGER) = CAST(vbap.POSNR AS INTEGER)
   LEFT JOIN (
       SELECT MATNR, SUM(LABST) AS total_stock FROM MARD GROUP BY MATNR
   ) stock ON stock.MATNR = vbap.MATNR
   WHERE CAST(vbap.VBELN AS INTEGER) = CAST('40' AS INTEGER)
     AND CAST(vbap.POSNR AS INTEGER) = CAST('10' AS INTEGER)
   LIMIT 1;

Before answering, mentally check: did I CAST every VBELN/POSNR? Did I aggregate
MARD, MAKT, VBEP? Did I COALESCE stock to 0? If any answer is "no", rewrite.

═════════════════ SCHEMA ═════════════════

{schema}
"""

_HUMAN_TEMPLATE = """\
Generate a SQLite SELECT query to validate the current state of a sales order item.

Parameters:
  - VBELN (order_id): {order_id}
  - POSNR (item_no) : {item_no}

Require these columns in the result (use these exact aliases):
  - material_name    (MAKT.MAKTX — English preferred, else any language; see rule 3)
  - quantity         (from VBAP.KWMENG)
  - delivery_status  (from VBUP.WBSTA)
  - delivery_date    (earliest VBEP.EDATU; see rule 3)
  - available_stock  (SUM of MARD.LABST across all storage locations, COALESCE to 0; see rule 3)
"""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    ("human",  _HUMAN_TEMPLATE),
])

# ---------------------------------------------------------------------------
# LLM factory (OpenRouter)
# ---------------------------------------------------------------------------

def _build_openrouter_llm(model_name: str, temperature: float) -> ChatOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    cfg = get_config()
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_key=api_key,
        openai_api_base=cfg.openrouter.base_url,
        default_headers={
            "HTTP-Referer": "https://github.com/daisysooyeon/SAP-ERP-AI-Agent",
            "X-Title": "SAP ERP AI Agent",
        },
    )


def _build_chain():
    """LLM 체인을 lazy하게 생성"""
    cfg = get_config()
    sql_cfg = cfg.models.worker_a_sql
    return _PROMPT | _build_openrouter_llm(sql_cfg.name, sql_cfg.temperature)


_chain = None


def _get_chain():
    global _chain
    if _chain is None:
        _chain = _build_chain()
    return _chain

# ---------------------------------------------------------------------------
# SQL 정제 헬퍼
# ---------------------------------------------------------------------------

def _clean_sql(raw: str) -> str:
    """LLM 응답에서 마크다운 코드 블록 제거 후 SQL만 추출"""
    # ```sql ... ``` 또는 ``` ... ``` 제거
    raw = re.sub(r"```(?:sql)?", "", raw, flags=re.IGNORECASE).strip()
    # 첫 번째 SELECT ~ 문장만 추출
    match = re.search(r"(SELECT\b.*)", raw, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else raw.strip()

# ---------------------------------------------------------------------------
# Hardcoded fallback query
# ---------------------------------------------------------------------------

def _hardcoded_query(order_id: str, item_no: str) -> str:
    # MARD / MAKT / VBEP 는 자재·아이템당 여러 행(저장위치·언어·일정라인)이므로
    # 직접 JOIN 후 LIMIT 1 하면 임의의(종종 NULL) 행을 집게 된다. 반드시 집계한다:
    #   available_stock : 전 저장위치 SUM(LABST)
    #   material_name   : 영어(SPRAS='E') 우선, 없으면 MIN(MAKTX)로 라틴/영문 우선
    #   delivery_date   : 가장 이른 일정라인 MIN(EDATU)
    return f"""
    SELECT
        makt.material_name              AS material_name,
        vbap.KWMENG                     AS quantity,
        vbup.WBSTA                      AS delivery_status,
        vbep.EDATU                      AS delivery_date,
        COALESCE(stock.total_stock, 0)  AS available_stock
    FROM VBAP vbap
    LEFT JOIN (
        SELECT MATNR,
               COALESCE(MAX(CASE WHEN SPRAS = 'E' THEN MAKTX END), MIN(MAKTX)) AS material_name
        FROM MAKT GROUP BY MATNR
    ) makt ON makt.MATNR = vbap.MATNR
    LEFT JOIN VBUP vbup
           ON CAST(vbup.VBELN AS INTEGER) = CAST(vbap.VBELN AS INTEGER)
          AND CAST(vbup.POSNR AS INTEGER) = CAST(vbap.POSNR AS INTEGER)
    LEFT JOIN (
        SELECT VBELN, POSNR, MIN(EDATU) AS EDATU FROM VBEP GROUP BY VBELN, POSNR
    ) vbep
           ON CAST(vbep.VBELN AS INTEGER) = CAST(vbap.VBELN AS INTEGER)
          AND CAST(vbep.POSNR AS INTEGER) = CAST(vbap.POSNR AS INTEGER)
    LEFT JOIN (
        SELECT MATNR, SUM(LABST) AS total_stock FROM MARD GROUP BY MATNR
    ) stock ON stock.MATNR = vbap.MATNR
    WHERE CAST(vbap.VBELN AS INTEGER) = CAST('{order_id}' AS INTEGER)
      AND CAST(vbap.POSNR AS INTEGER) = CAST('{item_no}'  AS INTEGER)
    LIMIT 1
    """

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_validation_query(order_id: str, item_no: str) -> tuple[str, str]:
    """
    LLM(primary → hardcoded) 순서로 SQLite 검증 쿼리를 생성하고 반환.

    Returns:
        (sql, strategy) — strategy는 다음 중 하나:
          "primary"            : LLM이 정상적으로 SQL을 생성
          "hardcoded_llm_fail" : LLM 호출/파싱 단계에서 예외 → hardcoded fallback 사용

        (run_validation_query는 여기에 더해 "hardcoded_zero_rows"를 별도로 사용한다.)
    """
    invoke_input = {
        "schema":   SCHEMA_CONTEXT,
        "order_id": order_id,
        "item_no":  item_no,
    }

    chain = _get_chain()

    # ── 1. Primary LLM ─────────────────────────────────────────────────────
    try:
        response = chain.invoke(invoke_input)
        sql = _clean_sql(response.content)
        logger.info("[text_to_sql] strategy=primary  | LLM 쿼리 생성 성공 (order=%s item=%s)",
                    order_id, item_no)
        logger.debug("[text_to_sql] SQL:\n%s", sql)
        return sql, "primary"
    except Exception as e:
        # LLM 실패는 운영상 주목해야 함 → ERROR 레벨 + 명시적 마커
        logger.error(
            "[text_to_sql] FALLBACK(llm_fail) | LLM 호출 실패 → hardcoded 쿼리 사용 "
            "(order=%s item=%s) | reason=%s: %s",
            order_id, item_no, type(e).__name__, e,
        )

    # ── 2. Hardcoded fallback ───────────────────────────────────────────────
    return _hardcoded_query(order_id, item_no), "hardcoded_llm_fail"


def _execute_query(sql: str) -> dict | None:
    """단일 SQL을 실행하고 첫 행을 dict로 반환 (없거나 오류 시 None)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(sql).fetchone()
        return dict(row) if row else None
    except sqlite3.Error as e:
        logger.error("[text_to_sql] SQLite 오류: %s\nSQL:\n%s", e, sql)
        return None
    finally:
        conn.close()


def run_validation_query(order_id: str, item_no: str) -> dict | None:
    """
    build_validation_query()로 생성된 쿼리를 SQLite에서 실행, 첫 번째 행을 dict로 반환.

    Primary(LLM) 쿼리가 0건을 반환하면 — LLM이 VBELN을 엄격 매칭(padded vs raw 불일치)
    하는 경우가 있으므로 — robust hardcoded 쿼리로 한 번 더 재시도한다. 둘 다 실패 시 None.

    fallback이 일어났는지 운영상 추적 가능하도록, 각 경로에서 명시적 WARNING 로그를 남긴다.
    """
    sql, strategy = build_validation_query(order_id, item_no)
    result = _execute_query(sql)

    if result is None and strategy == "primary":
        # LLM은 정상 호출됐는데 실행 결과가 0건 → JOIN/WHERE 규칙 누락이 의심됨
        logger.warning(
            "[text_to_sql] FALLBACK(primary_zero_rows) | LLM SQL이 0건 반환 → "
            "hardcoded 쿼리로 재시도 (order=%s item=%s)",
            order_id, item_no,
        )
        logger.debug("[text_to_sql] 0-row primary SQL:\n%s", sql)
        result = _execute_query(_hardcoded_query(order_id, item_no))
        if result is not None:
            logger.warning(
                "[text_to_sql] FALLBACK(primary_zero_rows) RECOVERED | hardcoded 쿼리가 "
                "row를 반환 → LLM 쿼리가 잘못 만들어졌다는 강한 신호 (order=%s item=%s)",
                order_id, item_no,
            )
        else:
            logger.warning(
                "[text_to_sql] FALLBACK(primary_zero_rows) STILL_EMPTY | hardcoded 쿼리도 "
                "0건 → 실제로 DB에 데이터가 없을 가능성 (order=%s item=%s)",
                order_id, item_no,
            )

    logger.info("[text_to_sql] 쿼리 실행 완료: strategy=%s result=%s", strategy, result)
    return result
