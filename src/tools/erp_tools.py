"""
src/tools/erp_tools.py
SQLite ERP DB 조회 / 업데이트 Tool
"""

import sqlite3
from pathlib import Path

DB_PATH = "data/sap_erp.db"


def query_sqlite(sql: str, params: tuple = ()) -> list[dict]:
    """SELECT 쿼리 실행 후 dict 목록 반환"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def update_sqlite(table: str, updates: dict, where: dict) -> int:
    """UPDATE 쿼리 실행 후 영향 행 수 반환"""
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    where_clause = " AND ".join(f"{k} = ?" for k in where)
    sql = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"
    params = tuple(updates.values()) + tuple(where.values())

    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.execute(sql, params)
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()
