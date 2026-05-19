"""
특정 router_test_cases_gen.json 항목만 재생성하는 일회성 스크립트.
사용: python -m src.data._regen_entries --ids r_023 r_029
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.data._llm_client import build_llm, invoke_with_retry
from src.data.generate_router_dataset import (
    _ACTION_PROMPT,
    _BOTH_PROMPT,
    _QA_PROMPT,
    _DUMMY_SQL_CASES,
    _is_quality_chunk,
    _load_chunks,
    _load_sql_cases,
    _pick_action,
)


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
        return {}

_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PATH = str(_ROOT / "data" / "eval" / "router_test_cases_gen.json")

_BAD_KEYWORDS = [
    "SAP SE or its affiliated companies",
    "forward-looking statements",
    "no obligation to pursue",
    "subject to various risks and uncertainties",
]


def _is_good_chunk(text: str) -> bool:
    # _is_quality_chunk에 이미 Learning Assessment, LESSON OBJECTIVES, UNIT TOC 필터 포함
    if not _is_quality_chunk(text):
        return False
    for kw in _BAD_KEYWORDS:
        if kw in text:
            return False
    return True


def regen_entries(ids: list[str], path: str, seed: int = 99, label: str = "ACTION_ONLY", db_path: str = "") -> None:
    data: list[dict] = json.loads(Path(path).read_text(encoding="utf-8"))
    id_map = {d["id"]: i for i, d in enumerate(data)}

    targets = []
    for rid in ids:
        if rid not in id_map:
            # 존재하지 않으면 새 항목으로 추가 (label 인자 필요)
            print(f"[INFO] {rid} not found — will create as new entry (requires --label)")
            targets.append({"id": rid, "_new": True})
            continue
        entry = data[id_map[rid]]
        if entry["label"] not in ("BOTH", "QA_ONLY", "ACTION_ONLY"):
            print(f"[WARN] {rid} is {entry['label']}, not supported — skipping")
            continue
        targets.append(entry)

    if not targets:
        print("재생성 대상 없음.")
        return

    all_chunks = [c for c in _load_chunks() if _is_good_chunk(c["text"])]
    rng = random.Random(seed)
    rng.shuffle(all_chunks)
    used_texts: set[str] = set()  # 이번 재생성에서 이미 쓴 청크 추적

    _root = Path(__file__).resolve().parent.parent.parent
    _db = db_path or str(_root / "data" / "sap_erp.db")
    sql_cases = _load_sql_cases(_db, 20, seed) or _DUMMY_SQL_CASES
    # order_id=40 과다 반복 방지: 기존 데이터에서 많이 쓰인 order 제외
    used_orders = [d["erp_evidence"]["order_id"] for d in data if d.get("erp_evidence")]
    from collections import Counter
    order_cnt = Counter(used_orders)
    diverse_cases = [c for c in sql_cases if order_cnt.get(str(c.get("order_id", "")), 0) < 3]
    if not diverse_cases:
        diverse_cases = sql_cases

    llm = build_llm()
    action_chain = _ACTION_PROMPT | llm
    both_chain   = _BOTH_PROMPT   | llm
    qa_chain     = _QA_PROMPT     | llm

    for entry in targets:
        entry_label = entry.get("label", label)
        is_new = entry.get("_new", False)
        bad_text = entry.get("rag_evidence", "") or ""
        chunk = next(
            (c for c in all_chunks if c["text"] != bad_text and c["text"] not in used_texts),
            next((c for c in all_chunks if c["text"] != bad_text), all_chunks[0]),
        )
        used_texts.add(chunk["text"])
        print(f"[{entry['id']}] 재생성 중 ({entry_label}) — chunk source: {chunk['source']}")

        if entry_label == "ACTION_ONLY":
            sql = diverse_cases[rng.randint(0, len(diverse_cases) - 1)]
            order_id    = str(sql.get("order_id", "4500012345"))
            item_no     = int(sql.get("item_no", 10))
            action      = _pick_action(order_id, item_no, rng)
            description = sql.get("description", "Standard sales order")
            raw = invoke_with_retry(
                action_chain,
                {"order_id": order_id, "item_no": item_no, "action": action, "description": description},
                label=f"regen/{entry['id']}",
            )
            if raw is None:
                print(f"  [FAIL] {entry['id']} LLM 호출 실패 — 건너뜀")
                continue
            email = raw.strip()
            if not email:
                print(f"  [FAIL] {entry['id']} 이메일 비어있음 — 건너뜀")
                continue
            new_entry = {
                "id":              entry["id"],
                "input":           email,
                "user_input":      email,
                "label":           "ACTION_ONLY",
                "expected_intent": "ACTION_ONLY",
                "erp_evidence": {
                    "order_id":        order_id,
                    "item_no":         item_no,
                    "action":          action,
                    "description":     description,
                    "expected_values": sql.get("expected_values"),
                },
                "rag_evidence":       None,
                "action_description": action,
                "qa_question":        None,
            }
            if is_new:
                data.append(new_entry)
            else:
                data[id_map[entry["id"]]] = new_entry

        elif entry_label == "QA_ONLY":
            raw = invoke_with_retry(
                qa_chain,
                {"chunk": chunk["text"]},
                label=f"regen/{entry['id']}",
            )
            if raw is None:
                print(f"  [FAIL] {entry['id']} LLM 호출 실패 — 건너뜀")
                continue
            parsed = _parse_json_response(raw)
            email = parsed.get("final_email", "").strip()
            if not email:
                print(f"  [FAIL] {entry['id']} 이메일 파싱 실패 — 건너뜀")
                continue
            idx = id_map[entry["id"]]
            data[idx]["input"]        = email
            data[idx]["user_input"]   = email
            data[idx]["rag_evidence"] = chunk["text"]
            data[idx]["qa_question"]  = parsed.get("draft_knowledge_question", "")

        else:  # BOTH
            ev = entry["erp_evidence"]
            raw = invoke_with_retry(
                both_chain,
                {
                    "order_id":    ev["order_id"],
                    "item_no":     ev["item_no"],
                    "action":      ev["action"],
                    "description": ev["description"],
                    "chunk":       chunk["text"],
                },
                label=f"regen/{entry['id']}",
            )
            if raw is None:
                print(f"  [FAIL] {entry['id']} LLM 호출 실패 — 건너뜀")
                continue
            parsed = _parse_json_response(raw)
            email = parsed.get("final_email", "").strip()
            if not email:
                print(f"  [FAIL] {entry['id']} 이메일 파싱 실패 — 건너뜀")
                continue
            idx = id_map[entry["id"]]
            data[idx]["input"]              = email
            data[idx]["user_input"]         = email
            data[idx]["rag_evidence"]       = chunk["text"]
            data[idx]["action_description"] = ev["action"]
            data[idx]["qa_question"]        = parsed.get("draft_knowledge_question", "")

        print(f"  [OK] {entry['id']} 교체 완료")

    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n저장 완료: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="+", required=True, help="재생성할 항목 ID (예: r_023 r_029)")
    parser.add_argument("--path", default=DEFAULT_PATH)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--label", default="ACTION_ONLY", help="새 항목 생성 시 사용할 레이블 (기본: ACTION_ONLY)")
    parser.add_argument("--db", default="", help="SQLite DB 경로 (기본: data/sap_erp.db)")
    args = parser.parse_args()
    regen_entries(args.ids, args.path, args.seed, args.label, args.db)
