"""
src/tools/text_to_sql.py
Text-to-SQL: dd03l 스키마 컨텍스트 기반 SQLite 검증 쿼리 생성

흐름:
  1. LLM (worker_a_sql / primary)  → SQL 생성
  2. LLM (worker_a_sql_fallback)   → primary 실패 시 재시도
  3. hardcoded fallback            → 양쪽 모두 실패 시
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

DB_PATH = "data/sap_erp.db"

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
  - VBELN is a 10-digit zero-padded TEXT string (e.g. '0000015353').
  - POSNR is an INTEGER (e.g. 10, 20).
  - EDATU is stored as TEXT in YYYYMMDD format.
"""

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a Text-to-SQL expert for SAP ERP data stored in SQLite.

Given the schema below and a request, generate a single valid SQLite SELECT query.

Rules:
1. Output ONLY the raw SQL — no markdown, no explanation, no code fences.
2. Always use LEFT JOIN so missing rows return NULL instead of no row.
3. End with LIMIT 1.
4. Use exact column and table names as shown in the schema.

Schema:
{schema}
"""

_HUMAN_TEMPLATE = """\
Generate a SQLite SELECT query to validate the current state of a sales order item.

Parameters:
  - VBELN (order_id): {order_id}
  - POSNR (item_no) : {item_no}

Require these columns in the result (use these exact aliases):
  - material_name    (from MAKT.MAKTX)
  - quantity         (from VBAP.KWMENG)
  - delivery_status  (from VBUP.WBSTA)
  - delivery_date    (from VBEP.EDATU)
  - available_stock  (from MARD.LABST)
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


def _build_chains():
    """primary / fallback LLM 체인을 lazy하게 생성"""
    cfg = get_config()
    sql_cfg      = cfg.models.worker_a_sql
    fallback_cfg = cfg.models.worker_a_sql_fallback

    primary  = _PROMPT | _build_openrouter_llm(sql_cfg.name,      sql_cfg.temperature)
    fallback = _PROMPT | _build_openrouter_llm(fallback_cfg.name, fallback_cfg.temperature)
    return primary, fallback


_primary_chain  = None
_fallback_chain = None


def _get_chains():
    global _primary_chain, _fallback_chain
    if _primary_chain is None:
        _primary_chain, _fallback_chain = _build_chains()
    return _primary_chain, _fallback_chain

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
    return f"""
    SELECT
        makt.MAKTX   AS material_name,
        vbap.KWMENG  AS quantity,
        vbup.WBSTA   AS delivery_status,
        vbep.EDATU   AS delivery_date,
        mard.LABST   AS available_stock
    FROM VBAP vbap
    LEFT JOIN MAKT makt ON vbap.MATNR = makt.MATNR AND makt.SPRAS = 'E'
    LEFT JOIN VBUP vbup ON vbap.VBELN = vbup.VBELN AND vbap.POSNR = vbup.POSNR
    LEFT JOIN VBEP vbep ON vbap.VBELN = vbep.VBELN AND vbap.POSNR = vbep.POSNR
    LEFT JOIN MARD mard ON vbap.MATNR = mard.MATNR
    WHERE vbap.VBELN = '{order_id}' AND vbap.POSNR = {item_no}
    LIMIT 1
    """

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_validation_query(order_id: str, item_no: str) -> tuple[str, str]:
    """
    LLM(primary → fallback → hardcoded) 순서로 SQLite 검증 쿼리를 생성하고 반환.

    Returns:
        (sql, strategy) — strategy는 "primary" | "fallback" | "hardcoded"
    """
    invoke_input = {
        "schema":   SCHEMA_CONTEXT,
        "order_id": order_id,
        "item_no":  item_no,
    }

    primary_chain, fallback_chain = _get_chains()

    # ── 1. Primary (qwen3-coder:free) ──────────────────────────────────────
    try:
        response = primary_chain.invoke(invoke_input)
        sql = _clean_sql(response.content)
        logger.info("[text_to_sql] primary LLM 쿼리 생성 성공")
        logger.debug("[text_to_sql] SQL:\n%s", sql)
        return sql, "primary"
    except Exception as e:
        logger.warning("[text_to_sql] primary LLM failed: %s --> trying fallback", e)

    # ── 2. Fallback (qwen3-coder-30b) ──────────────────────────────────────
    try:
        response = fallback_chain.invoke(invoke_input)
        sql = _clean_sql(response.content)
        logger.info("[text_to_sql] fallback LLM 쿼리 생성 성공")
        return sql, "fallback"
    except Exception as e:
        logger.warning("[text_to_sql] fallback LLM failed: %s --> using hardcoded query", e)

    # ── 3. Hardcoded fallback ───────────────────────────────────────────────
    logger.warning("[text_to_sql] 하드코딩 fallback 쿼리 사용")
    return _hardcoded_query(order_id, item_no), "hardcoded"


def run_validation_query(order_id: str, item_no: str) -> dict | None:
    """
    build_validation_query()로 생성된 쿼리를 SQLite에서 실행,
    첫 번째 행을 dict로 반환 (없으면 None).
    """
    sql, _ = build_validation_query(order_id, item_no)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(sql)
        row = cursor.fetchone()
        result = dict(row) if row else None
        logger.info("[text_to_sql] 쿼리 실행 완료: result=%s", result)
        return result
    except sqlite3.Error as e:
        logger.error("[text_to_sql] SQLite 오류: %s\nSQL:\n%s", e, sql)
        return None
    finally:
        conn.close()
