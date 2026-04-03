"""
src/tools/text_to_sql.py
Text-to-SQL: dd03l 스키마 컨텍스트 기반 SQLite 검증 쿼리 생성
"""

import sqlite3
from pathlib import Path

DB_PATH = "data/sap_erp.db"

# SAP ABAP 딕셔너리 기반 스키마 컨텍스트 (dd03l.csv에서 로드)
SCHEMA_CONTEXT = """
테이블: VBAK (영업 오더 헤더)
  VBELN: 영업 오더 번호 (PK), KUNNR: 고객코드, AUDAT: 오더일자

테이블: VBAP (영업 오더 아이템)
  VBELN: 오더번호 (FK), POSNR: 아이템번호, MATNR: 자재코드, KWMENG: 수량, NETPR: 단가

테이블: VBEP (납품 일정)
  VBELN: 오더번호, POSNR: 아이템번호, EDATU: 납기일, WMENG: 납품수량

테이블: VBUP (아이템 상태)
  VBELN: 오더번호, POSNR: 아이템번호, WBSTA: 출하상태 (A=미처리, B=부분출하, C=완전출하)

테이블: MARD (재고)
  MATNR: 자재코드, WERKS: 플랜트, LGORT: 저장위치, LABST: 가용재고
"""


def build_validation_query(order_id: str, item_no: str) -> str:
    """
    주어진 오더/아이템에 대한 검증용 SQLite 쿼리 반환
    TODO: Text-to-SQL LLM 연결 후 동적 쿼리 생성으로 교체
    """
    # 하드코딩된 기본 검증 쿼리 (Phase 3에서 LLM 생성으로 교체)
    return f"""
    SELECT
        vbap.KWMENG  AS quantity,
        vbup.WBSTA   AS delivery_status,
        vbep.EDATU   AS delivery_date,
        mard.LABST   AS available_stock
    FROM VBAP vbap
    LEFT JOIN VBUP vbup ON vbap.VBELN = vbup.VBELN AND vbap.POSNR = vbup.POSNR
    LEFT JOIN VBEP vbep ON vbap.VBELN = vbep.VBELN AND vbap.POSNR = vbep.POSNR
    LEFT JOIN MARD mard ON vbap.MATNR = mard.MATNR
    WHERE vbap.VBELN = '{order_id}' AND vbap.POSNR = '{item_no}'
    LIMIT 1
    """


def run_validation_query(order_id: str, item_no: str) -> dict | None:
    """검증 쿼리 실행 후 첫 번째 행 반환"""
    sql = build_validation_query(order_id, item_no)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(sql)
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()
