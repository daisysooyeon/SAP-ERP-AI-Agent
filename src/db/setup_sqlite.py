"""
src/db/setup_sqlite.py
Kaggle SAP Dataset CSV → SQLite DB 초기화 스크립트
"""

import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = "data/sap_erp.db"
RAW_DIR = "data/raw"

TABLE_CSV_MAP = {
    "VBAK": "vbak.csv",   # 영업 오더 헤더
    "VBAP": "vbap.csv",   # 영업 오더 아이템
    "VBEP": "vbep.csv",   # 납품 일정
    "VBUK": "vbuk.csv",   # 오더 상태 (헤더)
    "VBUP": "vbup.csv",   # 오더 상태 (아이템)
    "MAKT": "makt.csv",   # 자재 텍스트
    "MARD": "mard.csv",   # 재고 정보
}


def setup_db(raw_dir: str = RAW_DIR, db_path: str = DB_PATH) -> None:
    """CSV 파일을 읽어 SQLite에 테이블로 로드 (기존 테이블 덮어쓰기)"""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)

    for table_name, csv_file in TABLE_CSV_MAP.items():
        csv_path = Path(raw_dir) / csv_file
        if not csv_path.exists():
            print(f"[SKIP] {csv_file} 파일 없음")
            continue
        df = pd.read_csv(csv_path)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"[OK] {table_name}: {len(df)}행 로드")

    conn.close()
    print(f"[완료] SQLite DB 생성: {db_path}")


if __name__ == "__main__":
    setup_db()
