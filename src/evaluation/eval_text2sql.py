"""
src/evaluation/eval_text2sql.py
Text-to-SQL 성공률 평가
판정 기준: 실행 가능 여부 + 10초 이내 응답 + 예상 레코드 포함 여부
목표: 95% 이상
"""

import sqlite3
import signal

DB_PATH = "data/sap_erp.db"


def _timeout_handler(signum, frame):
    raise TimeoutError("SQL 실행 10초 초과")


def eval_sql(sql: str, expected: dict, db_path: str = DB_PATH) -> bool:
    """
    SQL 실행 성공 여부 반환
    Args:
        sql: 실행할 SQL 쿼리
        expected: {"order_id": str, ...} 예상 결과 포함 여부
        db_path: SQLite DB 경로
    """
    conn = sqlite3.connect(db_path)
    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(10)  # 10초 타임아웃
        cursor = conn.execute(sql)
        result = cursor.fetchall()
        signal.alarm(0)
        return expected.get("order_id", "") in str(result)
    except Exception:
        return False
    finally:
        conn.close()


if __name__ == "__main__":
    print("Text-to-SQL 평가 실행 예정")
