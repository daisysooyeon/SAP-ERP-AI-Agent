"""
scripts/db_backup.py
데모용 SQLite DB 스냅샷 백업/복원 도구

승인 플로우는 data/sap_erp.db 를 실제로 영구 변경한다(CHANGE_QTY → KWMENG,
CANCEL_ITEM → ABGRU 등). 데모를 반복하면 변경이 누적되므로, 깨끗한 원본을 한 번
백업해두고 데모 후 복원하는 용도로 사용한다.

사용법 (sap 가상환경에서):
    python scripts/db_backup.py backup     # 현재 DB를 스냅샷으로 저장 (최초 1회)
    python scripts/db_backup.py restore    # 스냅샷으로 DB 복원 (데모 후)
    python scripts/db_backup.py status     # 현재 DB vs 스냅샷 상태 확인

옵션:
    --force   backup 시 기존 스냅샷을 덮어쓴다 (기본은 보호)
    --db PATH      대상 DB 경로 직접 지정 (기본: configs.yaml의 paths.erp_db)
    --snapshot PATH 스냅샷 경로 직접 지정 (기본: <db>.demo_backup)
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 = 이 파일의 부모의 부모 (scripts/ 의 상위)
_ROOT = Path(__file__).resolve().parent.parent


def _default_db_path() -> Path:
    """configs.yaml의 paths.erp_db 를 우선 사용, 실패 시 data/sap_erp.db."""
    try:
        sys.path.insert(0, str(_ROOT))
        from src.config import get_config
        return (_ROOT / get_config().paths.erp_db).resolve()
    except Exception:
        return (_ROOT / "data" / "sap_erp.db").resolve()


def _fmt(path: Path) -> str:
    if not path.exists():
        return "(없음)"
    st = path.stat()
    mtime = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return f"{st.st_size:,} bytes  |  수정: {mtime}"


# 데모 샘플(web/index.html)이 실제로 수정하는 주문들 — 복원 전후 sanity check 대상.
# CANCEL_ITEM은 ABGRU(None→'ZZ'), CHANGE_QTY는 KWMENG가 바뀐다.
# (CHANGE_ADDR 17232/70은 human_loop에서 시뮬레이션이라 DB가 안 바뀜 — 대조용으로 함께 표시)
# 데모 샘플을 바꾸면 이 목록도 같이 맞춰야 한다.
_DEMO_PEEK_ROWS = [("15867", "20"), ("17232", "70")]


def _peek_row(db: Path) -> str:
    """데모 샘플이 건드리는 주문들의 KWMENG/ABGRU를 빠른 sanity check 로 보여준다."""
    if not db.exists():
        return ""
    try:
        c = sqlite3.connect(str(db))
        parts = []
        for vbeln, posnr in _DEMO_PEEK_ROWS:
            r = c.execute(
                "SELECT KWMENG, ABGRU FROM VBAP "
                "WHERE CAST(VBELN AS INTEGER)=CAST(? AS INTEGER) "
                "AND CAST(POSNR AS INTEGER)=CAST(? AS INTEGER)",
                (vbeln, posnr),
            ).fetchone()
            if r:
                parts.append(f"{vbeln}/{posnr} → KWMENG={r[0]!r} ABGRU={r[1]!r}")
        c.close()
        return "  |  ".join(parts)
    except sqlite3.Error:
        pass
    return ""


def cmd_backup(db: Path, snapshot: Path, force: bool) -> int:
    if not db.exists():
        print(f"[ERROR] DB가 없습니다: {db}")
        return 1
    if snapshot.exists() and not force:
        print(f"[중단] 스냅샷이 이미 존재합니다: {snapshot}")
        print("       기존 스냅샷을 보호합니다. 정말 덮어쓰려면 --force 를 붙이세요.")
        print(f"       현재 스냅샷: {_fmt(snapshot)}")
        return 1
    shutil.copy2(db, snapshot)
    print(f"[OK] 스냅샷 저장 완료")
    print(f"     원본   : {db}")
    print(f"     스냅샷 : {snapshot}")
    print(f"     {_peek_row(snapshot)}")
    return 0


def cmd_restore(db: Path, snapshot: Path) -> int:
    if not snapshot.exists():
        print(f"[ERROR] 스냅샷이 없습니다: {snapshot}")
        print("       먼저 'python scripts/db_backup.py backup' 으로 스냅샷을 만드세요.")
        return 1
    before = _peek_row(db)
    shutil.copy2(snapshot, db)
    after = _peek_row(db)
    print(f"[OK] 스냅샷으로 복원 완료")
    print(f"     스냅샷 → DB : {snapshot} → {db}")
    if before:
        print(f"     복원 전 : {before}")
    if after:
        print(f"     복원 후 : {after}")
    return 0


def cmd_status(db: Path, snapshot: Path) -> int:
    print("=== DB 스냅샷 상태 ===")
    print(f"DB     : {db}")
    print(f"         {_fmt(db)}")
    if _peek_row(db):
        print(f"         {_peek_row(db)}")
    print(f"스냅샷 : {snapshot}")
    print(f"         {_fmt(snapshot)}")
    if _peek_row(snapshot):
        print(f"         {_peek_row(snapshot)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="데모용 SQLite DB 백업/복원")
    parser.add_argument("command", choices=["backup", "restore", "status"])
    parser.add_argument("--force", action="store_true", help="backup 시 기존 스냅샷 덮어쓰기")
    parser.add_argument("--db", default=None, help="대상 DB 경로 (기본: configs.yaml)")
    parser.add_argument("--snapshot", default=None, help="스냅샷 경로 (기본: <db>.demo_backup)")
    args = parser.parse_args()

    db = Path(args.db).resolve() if args.db else _default_db_path()
    snapshot = Path(args.snapshot).resolve() if args.snapshot else db.with_suffix(db.suffix + ".demo_backup")

    if args.command == "backup":
        return cmd_backup(db, snapshot, args.force)
    if args.command == "restore":
        return cmd_restore(db, snapshot)
    return cmd_status(db, snapshot)


if __name__ == "__main__":
    raise SystemExit(main())
