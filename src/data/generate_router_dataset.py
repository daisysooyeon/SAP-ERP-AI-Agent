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
      "user_input": "...",        // 이메일 원문 (router·worker·e2e 평가 공통 입력)
      "label": "QA_ONLY",        // router 정답 레이블
      "erp_evidence": {...} | null,
      "rag_evidence": "..." | null,
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
import re
import sys
import time
from pathlib import Path
from dataclasses import asdict

from dotenv import load_dotenv
load_dotenv()

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from langchain_core.prompts import ChatPromptTemplate

from src.data._llm_client import build_llm, invoke_with_retry
from src.data.generate_text2sql_dataset import _sample_cases
from src.config import get_config

_ROOT = Path(__file__).resolve().parent.parent.parent
_LOG_DIR = _ROOT / "logs"
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

DEFAULT_OUTPUT = str(_ROOT / "data" / "eval" / "router_test_cases_gen.json")
DEFAULT_DB     = str(_ROOT / "data" / "sap_erp.db")

# ---------------------------------------------------------------------------
# ERP 액션 템플릿
# ---------------------------------------------------------------------------

# 각 템플릿이 action_type을 명시적으로 들고 있어 정규식 추측 없이 정확히 라벨링된다.
# CHANGE_QTY는 절대("change to {qty}") / 상대("reduce by {dec}") 두 모드를 구분한다.
_ERP_ACTIONS = [
    {"text": "Please change the quantity to {qty} units.", "type": "CHANGE_QTY", "qty_mode": "absolute"},
    {"text": "Kindly update the delivery date for this order.", "type": "CHANGE_DATE"},
    {"text": "We would like to cancel this order item.", "type": "CANCEL_ITEM"},
    {"text": "Please update the shipping address for this order.", "type": "CHANGE_ADDR"},
    {"text": "Reduce the order quantity by {dec} units.", "type": "CHANGE_QTY", "qty_mode": "relative"},
    {"text": "Please reschedule the delivery by two weeks.", "type": "CHANGE_DATE"},
    {"text": "Unblock the delivery for this order.", "type": "OTHER"},
    {"text": "Assign batch number B-{batch} to this item.", "type": "OTHER"},
    {"text": "Please set the payment terms to Net 60.", "type": "OTHER"},
    {"text": "Change the shipping method to express courier.", "type": "OTHER"},
]


def _make_action(rng: random.Random, expected_values: dict | None) -> tuple[str, str, str]:
    """랜덤 액션을 골라 (이메일 문장, action_type, expected_erp_action_status)를 반환한다.
    action_type은 템플릿이 명시적으로 들고 있고, status는 실제 운영 룰로 산출한다(아래)."""
    spec  = rng.choice(_ERP_ACTIONS)
    qty   = rng.randint(50, 500)
    dec   = rng.randint(10, 150)
    batch = rng.randint(10000, 99999)
    sentence = spec["text"].format(qty=qty, dec=dec, batch=batch)
    status   = _derive_status(spec, qty, dec, expected_values)
    return sentence, spec["type"], status


# ---------------------------------------------------------------------------
# Worker A 골든 상태 산출 — 운영 코드를 그대로 호출 (단일 출처, 드리프트 방지)
# ---------------------------------------------------------------------------

def _derive_status(spec: dict, qty: int, dec: int, expected_values: dict | None) -> str:
    """expected_erp_action_status를 운영 코드(resolve_quantity → check_business_rules)로
    그대로 산출한다. 정규식으로 룰을 재구현하지 않으므로 골든이 worker_a의 실제 판정과
    항상 1:1로 일치한다 — delta 재고검사·BLOCKED_INVALID_QTY·CANCEL+B 차단 등 모두 반영."""
    if not expected_values:
        return "BLOCKED_NO_DATA"

    from src.api.schemas import ERPActionRequest
    from src.graph.worker_a import resolve_quantity, check_business_rules

    action_type = spec["type"]
    kwargs: dict = {"order_id": "1", "item_no": "10", "action_type": action_type}
    if action_type == "CHANGE_QTY":
        if spec.get("qty_mode") == "relative":
            kwargs["quantity_change"] = -dec          # "reduce by {dec}"
        else:
            kwargs["new_quantity"] = qty              # "change to {qty}"
    elif action_type == "CHANGE_DATE":
        kwargs["new_date"] = "2030-01-01"

    try:
        req = ERPActionRequest(**kwargs)
    except Exception:
        return "PENDING_APPROVAL"

    # 운영과 동일: 상대수량 환산 → 비즈니스 룰. 둘 다 통과면 PENDING_APPROVAL.
    return (resolve_quantity(req, expected_values)
            or check_business_rules(req, expected_values)
            or "PENDING_APPROVAL")


# ---------------------------------------------------------------------------
# E3 다양성 타깃팅 — 각 BLOCKED_* 상태를 의도적으로 유도
# ---------------------------------------------------------------------------

def _forced_action(case: dict, target: str, rng: random.Random) -> tuple[str, str, str] | None:
    """특정 BLOCKED_* 상태를 유도하는 액션을 만들어 (sentence, action_type, status)를 반환한다.
    status는 실제 룰(_derive_status)로 산출하며, 의도(target)와 다르면 None(채택 안 함).
    - SHIPPED/PARTIAL : CANCEL_ITEM (케이스의 WBSTA가 각각 C/B여야 성립)
    - NO_STOCK        : 절대 증가량이 (현재+재고)를 넘게 → delta > 재고
    - INVALID_QTY     : 상대 감소량이 현재 수량을 넘게 → 음수
    """
    ev    = case.get("expected_values") or {}
    cur   = float(ev.get("quantity") or 0)
    stock = float(ev.get("available_stock") or 0)

    if target in ("SHIPPED", "PARTIAL"):
        spec = {"text": "We would like to cancel this order item.", "type": "CANCEL_ITEM"}
        status = _derive_status(spec, 0, 0, ev)
        want = "BLOCKED_SHIPPED" if target == "SHIPPED" else "BLOCKED_PARTIALLY_PROCESSED"
        return (spec["text"], "CANCEL_ITEM", status) if status == want else None

    if target == "NO_STOCK":
        qty  = int(cur + stock) + rng.randint(10, 100)
        spec = {"text": "Please change the quantity to {qty} units.", "type": "CHANGE_QTY", "qty_mode": "absolute"}
        sentence = spec["text"].format(qty=qty)
        status = _derive_status(spec, qty, 0, ev)
        return (sentence, "CHANGE_QTY", status) if status == "BLOCKED_NO_STOCK" else None

    if target == "INVALID_QTY":
        dec  = int(cur) + rng.randint(1, 50)
        spec = {"text": "Reduce the order quantity by {dec} units.", "type": "CHANGE_QTY", "qty_mode": "relative"}
        sentence = spec["text"].format(dec=dec)
        status = _derive_status(spec, 0, dec, ev)
        return (sentence, "CHANGE_QTY", status) if status == "BLOCKED_INVALID_QTY" else None

    return None


def _build_action_plan(
    sql_cases: list[dict],
    action_slots: list[dict],
    n_per_label: int,
    rng: random.Random,
) -> list[tuple[dict, tuple[str, str, str] | None]]:
    """ACTION_ONLY 슬롯 배정 계획. 앞쪽에 각 BLOCKED_* 상태를 유도하는 타깃 케이스를 넣고,
    나머지는 랜덤 액션(forced=None)으로 채운다. 길이는 n_per_label.
    데이터에 해당 시나리오(WBSTA=C/B 등)가 없으면 그 타깃은 자동 생략(best-effort).

    반환: [(sql_case, forced|None)] — forced = (sentence, action_type, expected_status)
    """
    def ds(c):    return (c.get("expected_values") or {}).get("delivery_status")
    def stock(c): return float((c.get("expected_values") or {}).get("available_stock") or 0)

    cases_C        = [c for c in sql_cases if ds(c) == "C"]
    cases_B        = [c for c in sql_cases if ds(c) == "B"]
    cases_nonC     = [c for c in sql_cases if ds(c) != "C"]
    cases_lowstock = sorted(cases_nonC, key=stock)   # 재고 적은 순 → NO_STOCK 수치를 현실적으로

    per_target = max(2, n_per_label // 12)

    plan: list[tuple[dict, tuple[str, str, str] | None]] = []

    def _add(case, target):
        forced = _forced_action(case, target, rng)
        if forced is not None:
            plan.append((case, forced))

    for c in rng.sample(cases_C,    min(per_target, len(cases_C))):    _add(c, "SHIPPED")
    for c in rng.sample(cases_B,    min(per_target, len(cases_B))):    _add(c, "PARTIAL")
    for c in cases_lowstock[:per_target]:                             _add(c, "NO_STOCK")
    for c in rng.sample(cases_nonC, min(per_target, len(cases_nonC))): _add(c, "INVALID_QTY")

    rng.shuffle(plan)
    plan = plan[:n_per_label]

    n_targeted = len(plan)
    for i in range(max(0, n_per_label - n_targeted)):
        plan.append((action_slots[i % len(action_slots)], None))

    logger.info(
        "ACTION_ONLY targeting: %d targeted slots (C=%d B=%d lowstock=%d) + %d random",
        n_targeted, len(cases_C), len(cases_B), len(cases_lowstock), n_per_label - n_targeted,
    )
    return plan


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
4. Output ONLY a valid JSON object with NO markdown formatting, using this structure:
{{
  "draft_knowledge_question": "The exact question you are going to ask",
  "final_email": "The full email text (3-5 sentences)"
}}
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
4. Output ONLY a valid JSON object with NO markdown formatting, using this structure:
{{
  "draft_knowledge_question": "The exact combined question you are going to ask",
  "final_email": "The full email text (3-5 sentences)"
}}
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
5. The "Order Scenario" is internal background for YOU only — NEVER mention internal database
   table names (MARD, VBAP, VBAK, etc.), JOINs, SQL, or technical scenario labels in the email.
   Write exactly as a real customer would (they don't know the backend).
6. Output ONLY the email text (3-5 sentences). No labels, no JSON, no explanation.
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
5. The "Order Scenario" is internal background for YOU only — NEVER mention internal database
   table names (MARD, VBAP, VBAK, etc.), JOINs, SQL, or technical scenario labels in the email.
   Write exactly as a real customer would (they don't know the backend).
6. Output ONLY a valid JSON object with NO markdown formatting, using this structure:
{{
  "draft_action_request": "The specific ERP request you are making",
  "draft_knowledge_question": "The specific how-to question you are asking",
  "final_email": "The full email text combining both (4-7 sentences)"
}}
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


# ---------------------------------------------------------------------------
# 청크 품질 필터
# ---------------------------------------------------------------------------

# 흔한 영어 단어(불용어 + 초고빈도 내용어) — OCR garble 판별용 앵커. 사전이 아니므로
# SAP 기술용어(Fiori/VBELN/ATP 등)는 오컷하지 않는다. 정상 영어 산문은 이 단어들이
# 35~50% 깔리지만(median 0.45), 유창 garble("epoteni Legareni Freedood")은 거의 없다(<0.15).
_COMMON_WORDS = frozenset((
    "the of to and a in is for that on with as are be this by an it or from at can you we have "
    "not will your they use which used when each all data system order customer document process "
    "create change between has more their if into other about its using based such may also where "
    "these new one two first see following example figure note must should within across after "
    "before during over under via per both then than only any our no but"
).split())


def _is_quality_chunk(text: str, min_length: int = 250) -> bool:
    """
    rag_evidence로 사용하기에 충분한 정보 밀도를 가진 청크인지 검사.
    아래 케이스를 필터링:
      - 너무 짧은 청크 (< min_length 문자)
      - SAP 저작권 URL만 있는 청크 (법적 고지문)
      - 학습 평가 답변/문제 청크 ("Learning Assessment")
      - 단원 도입부 헤더 ("LESSON OBJECTIVES")
      - TOC 패턴: "UNIT N" all-caps 또는 "Unit N\nM\n© Copyright" 형태
      - OCR garble: 알파벳 비율이 낮거나(숫자·기호 범벅 도표) 1~2글자 토큰이 과다(조각난 인식)
      - 유창 garble: 흔한 영어 단어 비율 < 0.15 (의미 없는 비단어 범벅 OCR; 실데이터 캘리브레이션)
    contextual_header("[Source: … | p.N]")가 본문에 prepend된 경우 헤더 줄을 제외한
    '본문'으로 판단한다 — 헤더가 길이·토큰 카운트를 부풀려 저품질 청크를 통과시키는 것 방지.
    """
    # contextual_header가 있으면 본문만 분리 (텍스트·OCR 청크 모두 "[Source:"로 시작)
    if text.startswith("[Source:"):
        sep = text.find("\n\n")
        text = text[sep + 2:] if sep != -1 else text.split("\n", 1)[-1]
    text = text.strip()
    if len(text) < min_length:
        return False
    if "https://www.sap.com/corporate/en/legal/copyright.html" in text:
        return False
    if "Learning Assessment" in text:
        return False
    if "LESSON OBJECTIVES" in text:
        return False
    # all-caps UNIT TOC 패턴: "UNIT 5\nControlling Sales Documents\nLesson 1\n..."
    if re.match(r'^UNIT\s+\d+', text):
        return False
    # TOC 패턴: 줄 대부분이 "Unit X", "Lesson N", 숫자, copyright만으로 구성
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    non_trivial = sum(
        1 for l in lines
        if len(l) > 15 and not l.isdigit()
        and not l.startswith('Unit ')
        and not l.startswith('Lesson')
        and not l.startswith('©')
        and 'All rights reserved' not in l
    )
    if non_trivial < 2:
        return False
    # ── OCR garble 휴리스틱 (정상 산문은 alpha≈0.7+, 단문토큰≈0.3 이라 여유 있게 통과) ──
    nonspace = [c for c in text if not c.isspace()]
    if nonspace and sum(c.isalpha() for c in nonspace) / len(nonspace) < 0.5:
        return False   # 숫자·기호 범벅 = 도표/표 OCR garble
    tokens = text.split()
    if tokens and sum(1 for t in tokens if len(t) <= 2) / len(tokens) > 0.5:
        return False   # 1~2글자 토큰 과다 = 조각난 OCR
    # ── 유창 garble 컷: 고빈도 영어 단어 비율 (실데이터 캘리브레이션 임계 0.15) ──
    words = re.findall(r"[A-Za-z]+", text)
    if words and sum(1 for w in words if w.lower() in _COMMON_WORDS) / len(words) < 0.15:
        return False   # 비단어 범벅 = 의미 없는 OCR garble
    return True


def _email_has_order(email: str, order_id: str) -> bool:
    """ERP grounding 검증: 생성된 이메일 본문에 실제 order_id가 들어있는지 확인.
    RAG 쪽 build_eval_ids의 'evidence ↔ 실제 청크' 매칭과 대칭되는 ERP 충실성 체크로,
    LLM이 프롬프트의 order_id를 무시/변형해 만든 이메일(r_013/r_023류 누수)을 걸러낸다."""
    return bool(order_id) and order_id in (email or "")


def load_chunks() -> list[dict]:
    """ChromaDB 컬렉션에 적재된 청크를 {text, source} 형태로 로드.
    (.get() 만 사용하므로 임베딩 모델 로딩 불필요 → chromadb 클라이언트 직접 사용)"""
    cfg = get_config()
    import chromadb
    try:
        client = chromadb.PersistentClient(path=cfg.paths.chroma_db)
        col = client.get_collection(cfg.rag.collection_name)
    except Exception as e:
        logger.warning(
            "ChromaDB 컬렉션 '%s' 로드 실패 (%s). 먼저 `python -m src.rag.ingest` 로 인제스트하세요.",
            cfg.rag.collection_name, e,
        )
        return []
    got = col.get(include=["documents", "metadatas"])
    docs = got.get("documents") or []
    metas = got.get("metadatas") or []
    return [
        {"text": d, "source": (m or {}).get("source", "")}
        for d, m in zip(docs, metas)
    ]


def sample_chunk_pairs(
    chunks: list[dict], n: int, rng: random.Random
) -> list[tuple[dict, dict]]:
    """멀티-청크 복합 QA용으로 청크 2개씩 n쌍 샘플링 (가능하면 서로 다른 source)."""
    if len(chunks) < 2 or n <= 0:
        return []
    pairs: list[tuple[dict, dict]] = []
    for _ in range(n):
        a = rng.choice(chunks)
        others = [c for c in chunks if c.get("source") != a.get("source")] or chunks
        b = rng.choice(others)
        pairs.append((a, b))
    return pairs


def _load_chunks() -> list[dict]:
    chunks = load_chunks()
    before = len(chunks)
    chunks = [c for c in chunks if _is_quality_chunk(c["text"])]
    logger.info("Chunk quality filter: %d → %d (removed %d)", before, len(chunks), before - len(chunks))
    return chunks


def _load_sql_cases(db_path: str, n: int, seed: int) -> list[dict]:
    """SQLite DB에서 text2sql 케이스를 샘플링합니다."""
    if not Path(db_path).exists():
        logger.warning("DB file not found: %s", db_path)
        return []
    try:
        per_scenario = max(2, n // 6 + 2)
        cases = _sample_cases(db_path, per_scenario, seed)
        records = [asdict(c) for c in cases]
        # negative case 제외
        return [r for r in records if r.get("order_id") and r["order_id"] != "9999999999"]
    except Exception as e:
        logger.warning("DB sampling failed: %s", e)
        return []




# ---------------------------------------------------------------------------
# 메인 생성 함수
# ---------------------------------------------------------------------------

def generate_router_dataset(
    n_per_label: int = 30,
    output_path: str = DEFAULT_OUTPUT,
    db_path: str = DEFAULT_DB,
    delay: float = 1.5,
    max_chunks: int | None = None,
    model_name: str | None = None,
    append: bool = False,
    seed: int = 42,
    multi_chunk_ratio: float = 0.0,
    enrich: bool = True,
) -> list[dict]:
    """
    Router + E2E 통합 레이블드 이메일 데이터셋을 생성합니다.

    QA_ONLY, ACTION_ONLY, BOTH 각 n_per_label개씩 균등 생성합니다.
    - QA_ONLY    : ChromaDB RAG 단일 청크 기반 개념/정책 질문 이메일
                   (multi_chunk_ratio > 0.0이면 해당 비율만큼 2-청크 복합 질문 포함)
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
    llm = build_llm(model_name=model_name)
    chunks = _load_chunks()

    if max_chunks is not None:
        chunks = chunks[:max_chunks]

    rng = random.Random(seed)

    sql_cases = _load_sql_cases(db_path, n_per_label * 4, seed)
    if not sql_cases:
        raise RuntimeError(
            f"No valid order cases loaded from DB: {db_path}\n"
            "ACTION_ONLY / BOTH generation requires real DB data.\n"
            "Run 'python -m src.db.setup_sqlite' first."
        )

    # 각 슬롯에 사용할 청크/sql 케이스 미리 결정
    shuffled_chunks = list(chunks)
    rng.shuffle(shuffled_chunks)
    shuffled_sql = list(sql_cases)
    rng.shuffle(shuffled_sql)

    qa_slots     = [shuffled_chunks[i % len(shuffled_chunks)] for i in range(n_per_label)]
    action_slots = [shuffled_sql[i % len(shuffled_sql)]                      for i in range(n_per_label)]
    both_sql     = [shuffled_sql[(i + n_per_label) % len(shuffled_sql)]      for i in range(n_per_label)]
    both_chunks  = [shuffled_chunks[(i + seed) % len(shuffled_chunks)] for i in range(n_per_label)]

    # QA_ONLY multi-chunk 슬롯 결정
    n_multi = max(0, int(n_per_label * multi_chunk_ratio))
    multi_indices = set(rng.sample(range(n_per_label), min(n_multi, n_per_label)))
    multi_pairs   = sample_chunk_pairs(chunks, len(multi_indices), rng)

    dataset: list[dict] = []
    _id_counter = 1  # skip된 케이스와 무관하게 연속 ID 보장

    def _parse_json_response(raw_text: str) -> dict:
        if not raw_text:
            return {}
        text = raw_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            logger.error("JSON parse failed: %s", raw_text)
            return {}

    # ── QA_ONLY ──────────────────────────────────────────────────────────────
    logger.info(
        "Generating QA_ONLY (target: %d, multi-chunk: %d)",
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
            
        parsed = _parse_json_response(raw)
        email = parsed.get("final_email", "").strip()
        if not email:
            continue
            
        qa_question = parsed.get("draft_knowledge_question", "")
            
        dataset.append({
            "id":                          f"r_{_id_counter:03d}",
            "user_input":                  email,
            "label":                       "QA_ONLY",
            "erp_evidence":                None,
            "rag_evidence":                rag_evidence,
            "qa_question":                 qa_question,
            "expected_action_type":        None,
            "expected_erp_action_status":  None,
        })
        _id_counter += 1

    logger.info("QA_ONLY done: %d", sum(1 for d in dataset if d["label"] == "QA_ONLY"))

    # ── QA_ONLY fill-up (retry failed slots) ─────────────────────────────────
    _fill_attempts = 0
    while sum(1 for d in dataset if d["label"] == "QA_ONLY") < n_per_label and _fill_attempts < n_per_label:
        _fill_attempts += 1
        chunk = rng.choice(shuffled_chunks)
        logger.info("  QA_ONLY fill-up [%d] source=%s", _fill_attempts, chunk["source"])
        raw = invoke_with_retry(qa_chain, {"chunk": chunk["text"]}, label="router/QA_ONLY/fill")
        time.sleep(delay)
        if raw is None:
            continue
        parsed = _parse_json_response(raw)
        email = parsed.get("final_email", "").strip()
        if not email:
            continue
        dataset.append({
            "id":                          f"r_{_id_counter:03d}",
            "user_input":                  email,
            "label":                       "QA_ONLY",
            "erp_evidence":                None,
            "rag_evidence":                chunk["text"],
            "qa_question":                 parsed.get("draft_knowledge_question", ""),
            "expected_action_type":        None,
            "expected_erp_action_status":  None,
        })
        _id_counter += 1
    if _fill_attempts:
        logger.info("QA_ONLY after fill-up: %d (fill attempts: %d)", sum(1 for d in dataset if d["label"] == "QA_ONLY"), _fill_attempts)

    # ── ACTION_ONLY ───────────────────────────────────────────────────────────
    logger.info("Generating ACTION_ONLY (target: %d)", n_per_label)
    action_chain = _ACTION_PROMPT | llm

    # E3 다양성 보장: 각 BLOCKED_* 상태를 유도하는 (케이스, 강제 액션)을 일부 슬롯에 배정.
    # 나머지는 랜덤 액션. 데이터에 해당 시나리오가 없으면 그 타깃은 자동 생략된다.
    action_plan = _build_action_plan(sql_cases, action_slots, n_per_label, rng)

    for i, (sql_case, forced) in enumerate(action_plan):
        order_id    = str(sql_case.get("order_id", "4500012345"))
        item_no     = int(sql_case.get("item_no", 10))
        if forced is not None:
            action, action_type, erp_status = forced
        else:
            action, action_type, erp_status = _make_action(rng, sql_case.get("expected_values"))
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
        if not email or not _email_has_order(email, order_id):
            if email:
                logger.warning("[ground] ACTION_ONLY order_id=%s 가 이메일에 없음 → 슬롯 폐기", order_id)
            continue
        dataset.append({
            "id":              f"r_{_id_counter:03d}",
            "user_input":      email,
            "label":           "ACTION_ONLY",
            "erp_evidence": {
                "order_id":        order_id,
                "item_no":         item_no,
                "action":          action,
                "description":     description,
                "expected_values": sql_case.get("expected_values"),
            },
            "rag_evidence":                None,
            "qa_question":                 None,
            "expected_action_type":        action_type,
            "expected_erp_action_status":  erp_status,
        })
        _id_counter += 1

    logger.info("ACTION_ONLY done: %d", sum(1 for d in dataset if d["label"] == "ACTION_ONLY"))

    # ── ACTION_ONLY fill-up (retry failed slots) ──────────────────────────────
    _fill_attempts = 0
    while sum(1 for d in dataset if d["label"] == "ACTION_ONLY") < n_per_label and _fill_attempts < n_per_label * 3:
        _fill_attempts += 1
        sql_case    = rng.choice(shuffled_sql)
        order_id    = str(sql_case.get("order_id"))
        item_no     = int(sql_case.get("item_no", 10))
        action, action_type, erp_status = _make_action(rng, sql_case.get("expected_values"))
        description = sql_case.get("description", "Standard sales order")
        logger.info("  ACTION_ONLY fill-up [%d] order=%s", _fill_attempts, order_id)
        raw = invoke_with_retry(
            action_chain,
            {"order_id": order_id, "item_no": item_no, "action": action, "description": description},
            label="router/ACTION_ONLY/fill",
        )
        time.sleep(delay)
        if raw is None:
            continue
        email = raw.strip()
        if not email or not _email_has_order(email, order_id):
            if email:
                logger.warning("[ground] ACTION_ONLY(fill) order_id=%s 가 이메일에 없음 → 슬롯 폐기", order_id)
            continue
        dataset.append({
            "id":              f"r_{_id_counter:03d}",
            "user_input":      email,
            "label":           "ACTION_ONLY",
            "erp_evidence": {
                "order_id":        order_id,
                "item_no":         item_no,
                "action":          action,
                "description":     description,
                "expected_values": sql_case.get("expected_values"),
            },
            "rag_evidence":                None,
            "qa_question":                 None,
            "expected_action_type":        action_type,
            "expected_erp_action_status":  erp_status,
        })
        _id_counter += 1
    if _fill_attempts:
        logger.info("ACTION_ONLY after fill-up: %d (fill attempts: %d)", sum(1 for d in dataset if d["label"] == "ACTION_ONLY"), _fill_attempts)

    # ── BOTH ──────────────────────────────────────────────────────────────────
    logger.info("Generating BOTH (target: %d)", n_per_label)
    both_chain = _BOTH_PROMPT | llm

    for i, (sql_case, chunk) in enumerate(zip(both_sql, both_chunks)):
        order_id    = str(sql_case.get("order_id", "4500012345"))
        item_no     = int(sql_case.get("item_no", 10))
        action, action_type, erp_status = _make_action(rng, sql_case.get("expected_values"))
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

        parsed = _parse_json_response(raw)
        email = parsed.get("final_email", "").strip()
        if not email or not _email_has_order(email, order_id):
            if email:
                logger.warning("[ground] BOTH order_id=%s 가 이메일에 없음 → 슬롯 폐기", order_id)
            continue
            
        qa_question = parsed.get("draft_knowledge_question", "")

        dataset.append({
            "id":              f"r_{_id_counter:03d}",
            "user_input":      email,
            "label":           "BOTH",
            "erp_evidence": {
                "order_id":        order_id,
                "item_no":         item_no,
                "action":          action,
                "description":     description,
                "expected_values": sql_case.get("expected_values"),
            },
            "rag_evidence":                chunk["text"],
            "qa_question":                 qa_question,
            "expected_action_type":        action_type,
            "expected_erp_action_status":  erp_status,
        })
        _id_counter += 1

    logger.info("BOTH done: %d", sum(1 for d in dataset if d["label"] == "BOTH"))

    # ── BOTH fill-up (retry failed slots) ─────────────────────────────────────
    _fill_attempts = 0
    while sum(1 for d in dataset if d["label"] == "BOTH") < n_per_label and _fill_attempts < n_per_label * 3:
        _fill_attempts += 1
        sql_case    = rng.choice(shuffled_sql)
        chunk       = rng.choice(shuffled_chunks)
        order_id    = str(sql_case.get("order_id"))
        item_no     = int(sql_case.get("item_no", 10))
        action, action_type, erp_status = _make_action(rng, sql_case.get("expected_values"))
        description = sql_case.get("description", "Standard sales order")
        logger.info("  BOTH fill-up [%d] order=%s source=%s", _fill_attempts, order_id, chunk["source"])
        raw = invoke_with_retry(
            both_chain,
            {"order_id": order_id, "item_no": item_no, "action": action, "description": description, "chunk": chunk["text"]},
            label="router/BOTH/fill",
        )
        time.sleep(delay)
        if raw is None:
            continue
        parsed = _parse_json_response(raw)
        email = parsed.get("final_email", "").strip()
        if not email or not _email_has_order(email, order_id):
            if email:
                logger.warning("[ground] BOTH(fill) order_id=%s 가 이메일에 없음 → 슬롯 폐기", order_id)
            continue
        dataset.append({
            "id":              f"r_{_id_counter:03d}",
            "user_input":      email,
            "label":           "BOTH",
            "erp_evidence": {
                "order_id":        order_id,
                "item_no":         item_no,
                "action":          action,
                "description":     description,
                "expected_values": sql_case.get("expected_values"),
            },
            "rag_evidence":                chunk["text"],
            "qa_question":                 parsed.get("draft_knowledge_question", ""),
            "expected_action_type":        action_type,
            "expected_erp_action_status":  erp_status,
        })
        _id_counter += 1
    if _fill_attempts:
        logger.info("BOTH after fill-up: %d (fill attempts: %d)", sum(1 for d in dataset if d["label"] == "BOTH"), _fill_attempts)

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
                logger.info("Appended %d to existing %d", len(dataset) - len(existing), len(existing))
            except Exception as e:
                logger.warning("Failed to read existing file — overwriting: %s", e)

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
    print("  Router + E2E Dataset Generation Results")
    print("=" * 60)
    print(f"  QA_ONLY     : {counts.get('QA_ONLY', 0)}  (multi-chunk: {n_multi_qa})")
    print(f"  ACTION_ONLY : {counts.get('ACTION_ONLY', 0)}")
    print(f"  BOTH        : {counts.get('BOTH', 0)}")
    print(f"  Total       : {len(dataset)}")
    print(f"  Output      : {output_path}")
    print("=" * 60)

    # ── enrich: 같은 경로에 evidence_chunk_id + golden_response 이어붙여 완성본 생성 ──
    # 기본 ON (--no-enrich로 끔). 1회 명령으로 평가용 완전 파일이 나온다.
    # build_eval_ids는 ChromaDB 인제스트가, golden_responses는 추가 LLM 호출이 필요하다.
    # 각 단계는 실패해도 흐름을 끊지 않고 경고만 남긴다(generate_all의 step 정책과 동일).
    if enrich:
        print("\n[enrich] build_eval_ids → evidence_chunk_id 부여 …")
        try:
            from src.rag.build_eval_ids import build_eval_ids
            build_eval_ids(Path(output_path), Path(output_path))
        except Exception as e:
            logger.warning("[enrich] build_eval_ids 실패 (ChromaDB 인제스트 확인) — 건너뜀: %s", e)
        print("[enrich] generate_golden_responses → golden_response 생성 …")
        try:
            from src.data.generate_golden_responses import generate_golden_responses
            generate_golden_responses(path=output_path, delay=delay, model_name=model_name)
        except Exception as e:
            logger.warning("[enrich] generate_golden_responses 실패 — 건너뜀: %s", e)
        # enrich 결과를 반환값에도 반영
        try:
            dataset = json.loads(Path(output_path).read_text(encoding="utf-8"))
        except Exception:
            pass

    return dataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Router + E2E 통합 레이블드 데이터셋 생성")
    parser.add_argument(
        "--n-per-label", type=int, default=30,
        help="클래스별 생성 목표 수 (기본: 30, 총 ≈ 90)",
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
        "--multi-chunk-ratio", type=float, default=0.0,
        help="QA_ONLY 슬롯 중 2-청크 복합 질문 비율 (0.0~1.0, 기본: 0.0)",
    )
    parser.add_argument(
        "--no-enrich", action="store_false", dest="enrich",
        help="보강 단계를 건너뜀(base 스키마만). 기본은 enrich 자동 실행 — 생성 후 같은 경로에 "
             "evidence_chunk_id + golden_response까지 붙여 완성본 생성. 스모크 테스트/저비용 시 사용.",
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
        enrich=args.enrich,
    )
