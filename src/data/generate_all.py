"""
src/data/generate_all.py
────────────────────────────────────────────────────────────────────────────
데이터셋 생성 오케스트레이터 CLI

모든 데이터셋을 한 번의 명령으로 순서대로 생성합니다.
router 단계가 e2e 데이터셋을 겸합니다 (QA_ONLY/ACTION_ONLY/BOTH 균등 생성).

실행 예시:
  # 전체 생성 (권장 실행 순서: text2sql → rag → router)
  python -m src.data.generate_all

  # 특정 타겟만 선택
  python -m src.data.generate_all --text2sql --rag
  python -m src.data.generate_all --router

  # 파라미터 조정
  python -m src.data.generate_all --n-per-label 15 --n-sql 10

  # 빠른 테스트 실행 (소량만 생성)
  python -m src.data.generate_all --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

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

# ---------------------------------------------------------------------------
# 기본 출력 경로
# ---------------------------------------------------------------------------

OUTPUTS = {
    "text2sql": str(_ROOT / "data" / "eval" / "text2sql_test_cases_gen.json"),
    "rag":      str(_ROOT / "data" / "eval" / "rag_test_cases.json"),
    "router":   str(_ROOT / "data" / "eval" / "router_test_cases_gen.json"),
}

# ---------------------------------------------------------------------------
# 생성 단계 실행 함수
# ---------------------------------------------------------------------------

def _step_text2sql(args: argparse.Namespace) -> bool:
    from src.data.generate_text2sql_dataset import generate_text2sql_dataset
    print("\n" + "━" * 70)
    print("  STEP 1/3 — Text-to-SQL Dataset Generation")
    print("━" * 70)
    try:
        generate_text2sql_dataset(
            n_samples=args.n_sql,
            output_path=args.out_text2sql,
            db_path=str(_ROOT / "data" / "sap_erp.db"),
            seed=args.seed,
            append=args.append,
        )
        return True
    except Exception as e:
        logger.error("text2sql generation failed: %s", e, exc_info=True)
        return False


def _step_rag(args: argparse.Namespace) -> bool:
    from src.data.generate_rag_dataset import generate_rag_dataset
    print("\n" + "━" * 70)
    print("  STEP 2/3 — RAG Dataset Generation (Chunk-based Q&A)")
    print("━" * 70)
    try:
        generate_rag_dataset(
            n_per_chunk=args.n_per_chunk,
            output_path=args.out_rag,
            delay=args.delay,
            verify=not args.no_verify,
            max_chunks=args.max_chunks,
            model_name=args.model,
            append=args.append,
        )
        return True
    except Exception as e:
        logger.error("RAG generation failed: %s", e, exc_info=True)
        return False


def _step_router(args: argparse.Namespace) -> bool:
    from src.data.generate_router_dataset import generate_router_dataset
    print("\n" + "━" * 70)
    print("  STEP 3/3 — Router + E2E Dataset Generation (QA_ONLY / ACTION_ONLY / BOTH)")
    print("━" * 70)
    try:
        generate_router_dataset(
            n_per_label=args.n_per_label,
            output_path=args.out_router,
            db_path=str(_ROOT / "data" / "sap_erp.db"),
            delay=args.delay,
            max_chunks=args.max_chunks,
            model_name=args.model,
            append=args.append,
            seed=args.seed,
        )
        return True
    except Exception as e:
        logger.error("Router generation failed: %s", e, exc_info=True)
        return False


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # --dry-run : 소량만 빠르게 테스트
    if args.dry_run:
        logger.info("Dry-run mode: executing each step with a small amount of data.")
        args.n_sql       = 10
        args.n_per_chunk = 1
        args.n_per_label = 30
        args.max_chunks  = 3
        args.no_verify   = True

    # 실행할 단계 결정 (플래그 없으면 전체 실행)
    run_all = not (args.text2sql or args.rag or args.router)

    steps: list[tuple[str, callable]] = []
    if run_all or args.text2sql:
        steps.append(("text2sql", _step_text2sql))
    if run_all or args.rag:
        steps.append(("rag",      _step_rag))
    if run_all or args.router:
        steps.append(("router",   _step_router))

    print("\n" + "=" * 70)
    print("  SAP ERP AI Agent Dataset Generation Pipeline")
    print(f"  Execution steps: {[s[0] for s in steps]}")
    print("=" * 70)

    t_start = time.perf_counter()
    results: dict[str, bool] = {}

    for name, fn in steps:
        results[name] = fn(args)
        if not results[name]:
            logger.warning("[%s] Step failed — proceeding to next step.", name)

    elapsed = time.perf_counter() - t_start

    # 최종 요약
    print("\n" + "=" * 70)
    print("  Final Result Summary")
    print("=" * 70)
    for name, ok in results.items():
        status = "✓ Success" if ok else "✗ Failed"
        print(f"  {name:<12}: {status}")
    print(f"  Total duration : {elapsed / 60:.1f} mins")
    print("=" * 70)

    # 출력 파일 경로 안내
    print("\n  Generated Files:")
    path_map = {
        "text2sql": args.out_text2sql,
        "rag":      args.out_rag,
        "router (= e2e)": args.out_router,
    }
    for label, path in path_map.items():
        name = label.split()[0]
        if name in results:
            exists = Path(path).exists()
            mark = "✓" if exists else "✗"
            print(f"  [{mark}] {label:<20}: {path}")


# ---------------------------------------------------------------------------
# CLI 정의
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAP ERP AI Agent 전체 데이터셋 생성 오케스트레이터",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python -m src.data.generate_all                      # 전체 실행
  python -m src.data.generate_all --dry-run            # 빠른 테스트 (소량)
  python -m src.data.generate_all --rag --n-per-chunk 5
  python -m src.data.generate_all --router --n-per-label 20
  python -m src.data.generate_all --text2sql --n-sql 30
        """,
    )

    # ── 단계 선택 ────────────────────────────────────────────────────────────
    step_group = parser.add_argument_group("실행 단계 선택 (미지정 시 전체 실행)")
    step_group.add_argument("--text2sql", action="store_true", help="Text-to-SQL 케이스 생성")
    step_group.add_argument("--rag",      action="store_true", help="RAG Q&A 케이스 생성")
    step_group.add_argument("--router",   action="store_true", help="Router + E2E 케이스 생성")

    # ── 생성 파라미터 ────────────────────────────────────────────────────────
    param_group = parser.add_argument_group("생성 파라미터")
    param_group.add_argument(
        "--n-per-chunk", type=int, default=1,
        help="RAG: 청크당 Q&A 생성 수 (기본: 1)",
    )
    param_group.add_argument(
        "--n-per-label", type=int, default=30,
        help="Router: 클래스(QA/ACTION/BOTH)별 생성 수 (기본: 30, 총 ≈ 90)",
    )
    param_group.add_argument(
        "--n-sql", type=int, default=10,
        help="Text-to-SQL: 시나리오당 샘플 수 (기본: 10)",
    )
    param_group.add_argument(
        "--max-chunks", type=int, default=100,
        help="RAG/Router: 최대 처리 청크 수 (기본: 100)",
    )
    param_group.add_argument(
        "--delay", type=float, default=1.5,
        help="API 호출 간격 초 (기본: 1.5)",
    )
    param_group.add_argument(
        "--seed", type=int, default=42,
        help="랜덤 시드 (기본: 42)",
    )
    param_group.add_argument(
        "--model", default=None,
        help="OpenRouter 모델명 오버라이드 (기본: configs.yaml worker_a)",
    )
    param_group.add_argument(
        "--no-verify", action="store_true",
        help="RAG Self-Verification 필터 비활성화 (빠르지만 품질 저하 가능)",
    )
    param_group.add_argument(
        "--append", action="store_true",
        help="기존 파일에 결과 추가 (기본: 새로 작성)",
    )
    param_group.add_argument(
        "--dry-run", action="store_true",
        help="소량만 실행하는 빠른 테스트 모드",
    )

    # ── 출력 경로 ────────────────────────────────────────────────────────────
    out_group = parser.add_argument_group("출력 경로 오버라이드")
    out_group.add_argument("--out-text2sql", default=OUTPUTS["text2sql"])
    out_group.add_argument("--out-rag",      default=OUTPUTS["rag"])
    out_group.add_argument("--out-router",   default=OUTPUTS["router"])

    return parser.parse_args()


if __name__ == "__main__":
    main()
