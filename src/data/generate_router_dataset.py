"""
src/data/generate_router_dataset.py
────────────────────────────────────────────────────────────────────────────
Router + E2E 통합 레이블드 데이터셋 생성기

흐름:
  1. QA_ONLY  : ChromaDB 청크에서 SAP 개념/정책 질문 이메일 생성
  2. ACTION_ONLY : SQLite DB의 실제 주문 데이터로 ERP 액션 이메일 생성
  3. BOTH     : 실제 주문 데이터 + RAG 청크를 결합한 복합 이메일 생성

각 클래스를 n_per_label개씩 균등 생성합니다.

출력 스키마 (router 평가 + e2e 평가 겸용):
  [
    {
      "id": "r_001",
      "input": "...",             // router 평가용 (user_input과 동일)
      "user_input": "...",        // e2e 평가용
      "label": "QA_ONLY",        // router 정답 레이블
      "expected_intent": "QA_ONLY",
      "erp_evidence": {...} | null,
      "rag_evidence": "..." | null,
      "action_description": "..." | null,
      "qa_question": "..." | null
    }
  ]

실행:
  python -m src.data.generate_router_dataset
  python -m src.data.generate_router_dataset --n-per-label 15 --output data/eval/router_test_cases_gen.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from dataclasses import asdict

from dotenv import load_dotenv
load_dotenv()

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_core.prompts import ChatPromptTemplate

from src.data._llm_client import build_llm, invoke_with_retry
from src.data.generate_rag_dataset import load_chunks
from src.data.generate_text2sql_dataset import _sample_cases

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
DEFAULT_OUTPUT = str(_ROOT / "data" / "eval" / "router_test_cases_gen.json")
DEFAULT_DB     = str(_ROOT / "data" / "sap_erp.db")

# ---------------------------------------------------------------------------
# ERP 액션 템플릿
# ---------------------------------------------------------------------------

_ERP_ACTIONS = [
    "Please change the quantity to {qty} units.",
    "Kindly update the delivery date for this order.",
    "We would like to cancel this order item.",
    "Please update the shipping address for this order.",
    "Reduce the order quantity by 50 units.",
    "Please reschedule the delivery by two weeks.",
    "Unblock the delivery for this order.",
    "Assign batch number B-{batch} to this item.",
    "Please set the payment terms to Net 60.",
    "Change the shipping method to express courier.",
]


def _pick_action(order_id: str, item_no: int, rng: random.Random) -> str:
    template = rng.choice(_ERP_ACTIONS)
    return template.format(qty=rng.randint(50, 500), batch=rng.randint(10000, 99999))


# ---------------------------------------------------------------------------
# 프롬프트 — 각 레이블별 단일 이메일 생성
# ---------------------------------------------------------------------------

_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
You are an expert at simulating realistic B2B SAP ERP email scenarios.

Write ONE email from a B2B customer or internal user asking a practical question about how to use the SAP ERP system or how a specific system process works.

Rules:
1. The email must NOT contain any specific ERP data modification or action requests (e.g., do not ask to change an order).
2. Focus ONLY on asking a "how-to" or system-related question directly inspired by the provided SAP manual chunk. Make it sound like a user trying to understand or use the system.
3. Use formal business English.
4. Output ONLY the email text (3-5 sentences). No labels, no JSON, no explanation.
"""),
    ("human", "SAP Manual Chunk (for inspiration):\n\n{chunk}"),
])

_QA_MULTI_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
You are an expert at simulating realistic B2B SAP ERP email scenarios.

Write ONE email from a B2B customer or internal user asking a question that naturally spans BOTH of the provided SAP manual passages.

Rules:
1. The email must NOT contain any ERP data modification or action requests.
2. The question should require information from BOTH passages to answer fully — not answerable from either alone.
3. Use formal business English.
4. Output ONLY the email text (3-5 sentences). No labels, no JSON, no explanation.
"""),
    ("human", "SAP Manual Passage A:\n\n{chunk_a}\n\n---\n\nSAP Manual Passage B:\n\n{chunk_b}"),
])

_ACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
You are writing a realistic B2B customer email requesting a specific SAP ERP action.

Rules:
1. Reference the actual order number and item number naturally in the email.
2. Make the action request clear and specific.
3. Do NOT include any policy or concept questions.
4. Use formal business English.
5. Output ONLY the email text (3-5 sentences). No labels, no JSON, no explanation.
"""),
    ("human", """\
Order ID: {order_id}
Item No: {item_no}
Requested Action: {action}
Order Scenario: {description}
"""),
])

_BOTH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """\
You are writing a realistic B2B customer email that naturally combines:
1. A specific SAP ERP action request (referencing real order details)
2. A practical system usage or "how-to" question about the SAP ERP system (inspired by the provided content)

Rules:
1. Both the action request and the system usage question must appear naturally in a single email.
2. Reference the actual order number and item number for the action request.
3. Make the system question directly related to how a user interacts with the ERP based on the chunk.
4. Use formal business English.
5. Output ONLY the email text (4-7 sentences). No labels, no JSON, no explanation.
"""),
    ("human", """\
ERP Action Request:
  Order ID: {order_id} / Item No: {item_no}
  Action: {action}
  Order Scenario: {description}

SAP Content (for the system usage question):
{chunk}
"""),
])


# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------

def _load_chunks() -> list[dict]:
    return load_chunks()


def _load_sql_cases(db_path: str, n: int, seed: int) -> list[dict]:
    """SQLite DB에서 text2sql 케이스를 샘플링합니다."""
    if not Path(db_path).exists():
        logger.warning("DB 파일 없음 — ACTION_ONLY/BOTH에 더미 케이스 사용: %s", db_path)
        return []
    try:
        per_scenario = max(2, n // 6 + 2)
        cases = _sample_cases(db_path, per_scenario, seed)
        records = [asdict(c) for c in cases]
        # negative case 제외
        return [r for r in records if r.get("order_id") and r["order_id"] != "9999999999"]
    except Exception as e:
        logger.warning("DB 샘플링 실패: %s", e)
        return []


_DUMMY_SQL_CASES = [
    {"order_id": "4500012345", "item_no": 10,
     "description": "Standard sales order", "expected_values": None},
    {"order_id": "4500067890", "item_no": 20,
     "description": "Order with partial delivery", "expected_values": None},
    {"order_id": "4500099001", "item_no": 10,
     "description": "Order pending goods movement", "expected_values": None},
]


# ---------------------------------------------------------------------------
# 메인 생성 함수
# ---------------------------------------------------------------------------

def generate_router_dataset(
    n_per_label: int = 10,
    output_path: str = DEFAULT_OUTPUT,
    db_path: str = DEFAULT_DB,
    delay: float = 1.5,
    max_chunks: int | None = None,
    model_name: str | None = None,
    append: bool = False,
    seed: int = 42,
    multi_chunk_ratio: float = 0.3,
) -> list[dict]:
    """
    Router + E2E 통합 레이블드 이메일 데이터셋을 생성합니다.

    QA_ONLY, ACTION_ONLY, BOTH 각 n_per_label개씩 균등 생성합니다.
    - QA_ONLY    : ChromaDB RAG 청크 기반 개념/정책 질문 이메일
                   (multi_chunk_ratio 비율만큼 2-청크 복합 질문 포함)
    - ACTION_ONLY: SQLite DB 실제 주문 데이터 기반 ERP 액션 이메일
    - BOTH       : 실제 주문 데이터 + RAG 청크 결합 복합 이메일

    Parameters
    ----------
    n_per_label       : 클래스별 생성 목표 수 (총 ≈ n_per_label × 3)
    output_path       : 출력 JSON 경로
    db_path           : SQLite DB 경로 (sap_erp.db)
    delay             : API 호출 간격(초)
    max_chunks        : 사용할 최대 청크 수
    model_name        : OpenRouter 모델명 오버라이드
    append            : 기존 파일에 추가 여부
    seed              : 랜덤 시드
    multi_chunk_ratio : QA_ONLY 슬롯 중 2-청크 복합 질문 비율 (0.0~1.0, 기본 0.3)
    """
    from src.data.generate_rag_dataset import sample_chunk_pairs

    llm = build_llm(model_name=model_name)
    chunks = _load_chunks()

    if max_chunks is not None:
        chunks = chunks[:max_chunks]

    rng = random.Random(seed)

    sql_cases = _load_sql_cases(db_path, n_per_label * 4, seed) or _DUMMY_SQL_CASES

    # 각 슬롯에 사용할 청크/sql 케이스 미리 결정
    shuffled_chunks = list(chunks)
    rng.shuffle(shuffled_chunks)
    shuffled_sql = list(sql_cases)
    rng.shuffle(shuffled_sql)

    qa_slots     = [shuffled_chunks[i % len(shuffled_chunks)] for i in range(n_per_label)]
    action_slots = [shuffled_sql[i % len(shuffled_sql)]       for i in range(n_per_label)]
    both_sql     = [shuffled_sql[i % len(shuffled_sql)]        for i in range(n_per_label)]
    both_chunks  = [shuffled_chunks[(i + seed) % len(shuffled_chunks)] for i in range(n_per_label)]

    # QA_ONLY multi-chunk 슬롯 결정
    n_multi = max(0, int(n_per_label * multi_chunk_ratio))
    multi_indices = set(rng.sample(range(n_per_label), min(n_multi, n_per_label)))
    multi_pairs   = sample_chunk_pairs(chunks, len(multi_indices), rng)

    dataset: list[dict] = []

    # ── QA_ONLY ──────────────────────────────────────────────────────────────
    logger.info(
        "QA_ONLY 생성 시작 (목표: %d개, 그 중 multi-chunk: %d개)",
        n_per_label, len(multi_indices),
    )
    qa_chain       = _QA_PROMPT | llm
    qa_multi_chain = _QA_MULTI_PROMPT | llm
    multi_pair_iter = iter(multi_pairs)

    for i, chunk in enumerate(qa_slots):
        if i in multi_indices:
            pair = next(multi_pair_iter, None)
            if pair is None:
                pair = (chunk, shuffled_chunks[(i + 1) % len(shuffled_chunks)])
            chunk_a, chunk_b = pair
            logger.info(
                "  QA_ONLY [%d/%d] MULTI src=%s+%s",
                i + 1, n_per_label, chunk_a["source"], chunk_b["source"],
            )
            raw = invoke_with_retry(
                qa_multi_chain,
                {"chunk_a": chunk_a["text"], "chunk_b": chunk_b["text"]},
                label="router/QA_MULTI",
            )
            rag_evidence = [chunk_a["text"], chunk_b["text"]]
        else:
            logger.info("  QA_ONLY [%d/%d] source=%s", i + 1, n_per_label, chunk["source"])
            raw = invoke_with_retry(
                qa_chain,
                {"chunk": chunk["text"]},
                label="router/QA_ONLY",
            )
            rag_evidence = chunk["text"]

        time.sleep(delay)
        if raw is None:
            continue
        email = raw.strip()
        if not email:
            continue
        dataset.append({
            "id":                f"r_{len(dataset)+1:03d}",
            "input":             email,
            "user_input":        email,
            "label":             "QA_ONLY",
            "expected_intent":   "QA_ONLY",
            "erp_evidence":      None,
            "rag_evidence":      rag_evidence,
            "action_description": None,
            "qa_question":       email[:300],
        })

    logger.info("QA_ONLY 생성 완료: %d개", sum(1 for d in dataset if d["label"] == "QA_ONLY"))

    # ── ACTION_ONLY ───────────────────────────────────────────────────────────
    logger.info("ACTION_ONLY 생성 시작 (목표: %d개)", n_per_label)
    action_chain = _ACTION_PROMPT | llm

    for i, sql_case in enumerate(action_slots):
        order_id    = str(sql_case.get("order_id", "4500012345"))
        item_no     = int(sql_case.get("item_no", 10))
        action      = _pick_action(order_id, item_no, rng)
        description = sql_case.get("description", "Standard sales order")

        logger.info("  ACTION_ONLY [%d/%d] order=%s", i + 1, n_per_label, order_id)
        raw = invoke_with_retry(
            action_chain,
            {
                "order_id":    order_id,
                "item_no":     item_no,
                "action":      action,
                "description": description,
            },
            label="router/ACTION_ONLY",
        )
        time.sleep(delay)
        if raw is None:
            continue
        email = raw.strip()
        if not email:
            continue
        dataset.append({
            "id":              f"r_{len(dataset)+1:03d}",
            "input":           email,
            "user_input":      email,
            "label":           "ACTION_ONLY",
            "expected_intent": "ACTION_ONLY",
            "erp_evidence": {
                "order_id":        order_id,
                "item_no":         item_no,
                "action":          action,
                "description":     description,
                "expected_values": sql_case.get("expected_values"),
            },
            "rag_evidence":       None,
            "action_description": action,
            "qa_question":        None,
        })

    logger.info("ACTION_ONLY 생성 완료: %d개", sum(1 for d in dataset if d["label"] == "ACTION_ONLY"))

    # ── BOTH ──────────────────────────────────────────────────────────────────
    logger.info("BOTH 생성 시작 (목표: %d개)", n_per_label)
    both_chain = _BOTH_PROMPT | llm

    for i, (sql_case, chunk) in enumerate(zip(both_sql, both_chunks)):
        order_id    = str(sql_case.get("order_id", "4500012345"))
        item_no     = int(sql_case.get("item_no", 10))
        action      = _pick_action(order_id, item_no, rng)
        description = sql_case.get("description", "Standard sales order")

        logger.info("  BOTH [%d/%d] order=%s source=%s", i + 1, n_per_label, order_id, chunk["source"])
        raw = invoke_with_retry(
            both_chain,
            {
                "order_id":    order_id,
                "item_no":     item_no,
                "action":      action,
                "description": description,
                "chunk":       chunk["text"],
            },
            label="router/BOTH",
        )
        time.sleep(delay)
        if raw is None:
            continue
        email = raw.strip()
        if not email:
            continue
        dataset.append({
            "id":              f"r_{len(dataset)+1:03d}",
            "input":           email,
            "user_input":      email,
            "label":           "BOTH",
            "expected_intent": "BOTH",
            "erp_evidence": {
                "order_id":        order_id,
                "item_no":         item_no,
                "action":          action,
                "description":     description,
                "expected_values": sql_case.get("expected_values"),
            },
            "rag_evidence":       chunk["text"],
            "action_description": action,
            "qa_question":        None,
        })

    logger.info("BOTH 생성 완료: %d개", sum(1 for d in dataset if d["label"] == "BOTH"))

    # ── append 처리 ───────────────────────────────────────────────────────────
    if append:
        existing_path = Path(output_path)
        if existing_path.exists():
            try:
                existing = json.loads(existing_path.read_text(encoding="utf-8"))
                max_id = max(
                    (int(e["id"].split("_")[1]) for e in existing if "id" in e),
                    default=0,
                )
                for idx, item in enumerate(dataset, start=max_id + 1):
                    item["id"] = f"r_{idx:03d}"
                dataset = existing + dataset
                logger.info("기존 %d개에 %d개 추가", len(existing), len(dataset) - len(existing))
            except Exception as e:
                logger.warning("기존 파일 읽기 실패 — 새로 작성: %s", e)

    # ── 저장 ──────────────────────────────────────────────────────────────────
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── 통계 ──────────────────────────────────────────────────────────────────
    from collections import Counter
    counts = Counter(d["label"] for d in dataset)
    n_multi_qa = sum(
        1 for d in dataset
        if d["label"] == "QA_ONLY" and isinstance(d.get("rag_evidence"), list)
    )
    print("\n" + "=" * 60)
    print("  Router + E2E 데이터셋 생성 결과")
    print("=" * 60)
    print(f"  QA_ONLY       : {counts.get('QA_ONLY', 0)}  (그 중 multi-chunk: {n_multi_qa})")
    print(f"  ACTION_ONLY   : {counts.get('ACTION_ONLY', 0)}")
    print(f"  BOTH          : {counts.get('BOTH', 0)}")
    print(f"  합계           : {len(dataset)}")
    print(f"  출력 파일      : {output_path}")
    print("=" * 60)

    return dataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Router + E2E 통합 레이블드 데이터셋 생성")
    parser.add_argument(
        "--n-per-label", type=int, default=10,
        help="클래스별 생성 목표 수 (기본: 10, 총 ≈ 30)",
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
        "--delay", type=float, default=1.5,
        help="API 호출 간격 초 (기본: 1.5)",
    )
    parser.add_argument(
        "--max-chunks", type=int, default=None,
        help="사용할 최대 청크 수 (기본: 전체)",
    )
    parser.add_argument(
        "--model", default=None,
        help="OpenRouter 모델명 오버라이드",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="기존 출력 파일에 결과를 추가",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="랜덤 시드 (기본: 42)",
    )
    parser.add_argument(
        "--multi-chunk-ratio", type=float, default=0.3,
        help="QA_ONLY 슬롯 중 2-청크 복합 질문 비율 (0.0~1.0, 기본: 0.3)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_router_dataset(
        n_per_label=args.n_per_label,
        output_path=args.output,
        db_path=args.db,
        delay=args.delay,
        max_chunks=args.max_chunks,
        model_name=args.model,
        append=args.append,
        seed=args.seed,
        multi_chunk_ratio=args.multi_chunk_ratio,
    )
