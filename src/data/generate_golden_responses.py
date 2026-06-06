"""
src/data/generate_golden_responses.py

router_test_cases_gen.json의 각 케이스에 Samsung SDS 물류팀 형식 템플릿으로
`golden_response`(정답 이메일 출력)를 채워넣는 결정론적 생성기.

LLM을 호출하지 않고 stdlib만 사용한다 — 케이스의 기존 필드(label /
action_description / qa_question / erp_evidence / rag_evidence)에서
구체적 내용을 직접 조합한다. 이렇게 만든 golden_response는 eval_e2e가
실제 모델 출력과 비교할 ground-truth 답안으로 쓰인다.

템플릿 구조 (요청 사양 — 영어 본문):

    Dear {first_name or "Customer"},

    Hello,

    This is the Samsung SDS Logistics Team.

    After our internal review of the matter you raised in your email,
    {specific content per case}

    Please feel free to contact us anytime for any related inquiries.

    Thank you.

실행:
    python -m src.data.generate_golden_responses
    python -m src.data.generate_golden_responses --in data/eval/router_test_cases_gen.json \\
                                                  --out data/eval/router_test_cases_gen.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 템플릿 — 인사·서명 고정. 본문(body)만 케이스별로 채워진다.
# ---------------------------------------------------------------------------

_TEMPLATE = (
    "Dear {greeting_name},\n"
    "\n"
    "Hello,\n"
    "\n"
    "This is the Samsung SDS Logistics Team.\n"
    "\n"
    "After our internal review of the matter you raised in your email,\n"
    "{body}\n"
    "\n"
    "Please feel free to contact us anytime for any related inquiries.\n"
    "\n"
    "Thank you."
)


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def _summarize_rag(rag_evidence: str | None, max_chars: int = 350) -> str:
    """rag_evidence를 답변 본문에 어울리는 형태로 정제 — 줄바꿈/페이지 표시 제거 후
    문장 단위로 자르고 max_chars 이내로 축약. LLM을 안 쓰는 결정론적 요약."""
    if not rag_evidence:
        return ""

    text = rag_evidence.replace("\n", " ").strip()
    # "Unit 12: ..." / "© Copyright" / "Page 158" 같은 출처 메타 제거
    text = re.sub(r"©\s*Copyright[^.]*\.?", "", text)
    text = re.sub(r"All rights reserved\.?", "", text)
    text = re.sub(r"Unit\s+\d+:[^.]*\.?", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()

    if len(text) <= max_chars:
        return text

    # 문장 경계로 자르기
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out, total = [], 0
    for s in sentences:
        if total + len(s) > max_chars and out:
            break
        out.append(s)
        total += len(s) + 1
    return " ".join(out).rstrip() + (" …" if total < len(text) else "")


def _action_body(action_description: str | None, erp_evidence: dict | None) -> str:
    """ACTION_ONLY (또는 BOTH의 action 파트) 본문 — 처리 결과 요약."""
    action = (action_description or "").strip().rstrip(".")
    ev     = erp_evidence or {}
    order  = ev.get("order_id", "")
    item   = ev.get("item_no", "")

    target = ""
    if order and item:
        target = f" on order {order}, item {item}"
    elif order:
        target = f" on order {order}"

    if action:
        return (
            f"we have processed your request: {action}{target}.\n"
            f"\n"
            f"The change has been applied successfully in our system, and you will see "
            f"it reflected on the relevant documents shortly."
        )
    return (
        f"we have processed your request{target}.\n"
        f"\n"
        f"The change has been applied successfully in our system."
    )


def _qa_body(qa_question: str | None, rag_evidence: str | None) -> str:
    """QA_ONLY (또는 BOTH의 QA 파트) 본문 — 질문 인용 + 근거 요약."""
    q = (qa_question or "").strip().rstrip("?")
    summary = _summarize_rag(rag_evidence)

    if not summary:
        return f"regarding your question — {q}? — we will get back to you with the relevant details shortly."

    if q:
        return (
            f"regarding your question — {q}? — please find the answer below.\n"
            f"\n"
            f"{summary}"
        )
    return summary


def _both_body(action_description, erp_evidence, qa_question, rag_evidence) -> str:
    """BOTH 본문 — action 처리 결과 + 질문 답변. _qa_body가 이미 'regarding'
    으로 시작하므로 별도 연결구는 빈 줄 두 개만 둔다."""
    return (
        _action_body(action_description, erp_evidence)
        + "\n\n"
        + "On your additional question, "
        + _qa_body(qa_question, rag_evidence)
    )


# ---------------------------------------------------------------------------
# 케이스별 골든 응답 생성
# ---------------------------------------------------------------------------

_NAME_RE = re.compile(
    r"(?:^|\n)(?:From|Best regards,|Sincerely,|Regards,|Thanks,?)\s*\n?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    re.MULTILINE,
)


def _extract_sender_first_name(email_text: str) -> str:
    """이메일 서명에서 발신자 이름의 first name을 보수적으로 추출.
    placeholder([Your Name], [Name])는 무시. 못 찾으면 빈 문자열."""
    if not email_text:
        return ""

    m = _NAME_RE.search(email_text)
    if not m:
        return ""
    full = m.group(1).strip()
    if re.search(r"\[.*\]", full) or full.lower() in {"your name", "name"}:
        return ""
    return full.split()[0]


def build_golden_response(case: dict) -> str:
    """케이스 하나에 대한 golden_response 문자열을 생성."""
    label = case.get("label", "")
    sender_first_name = _extract_sender_first_name(
        case.get("user_input") or case.get("input") or ""
    )
    greeting_name = sender_first_name or "Customer"

    if label == "ACTION_ONLY":
        body = _action_body(case.get("action_description"), case.get("erp_evidence"))
    elif label == "QA_ONLY":
        body = _qa_body(case.get("qa_question"), case.get("rag_evidence"))
    elif label == "BOTH":
        body = _both_body(
            case.get("action_description"),
            case.get("erp_evidence"),
            case.get("qa_question"),
            case.get("rag_evidence"),
        )
    else:
        body = "we have noted your message and will follow up shortly with the relevant details."

    return _TEMPLATE.format(greeting_name=greeting_name, body=body)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def populate(in_path: Path, out_path: Path, overwrite: bool = False) -> int:
    """in_path를 읽어 각 케이스에 golden_response를 채우고 out_path에 저장.
    overwrite=False면 이미 비어있지 않은 golden_response는 건드리지 않음."""
    cases: list[dict[str, Any]] = json.loads(in_path.read_text(encoding="utf-8"))
    changed = 0
    for c in cases:
        if not overwrite and c.get("golden_response"):
            continue
        c["golden_response"] = build_golden_response(c)
        changed += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(cases, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return changed


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    default_in = repo_root / "data" / "eval" / "router_test_cases_gen.json"

    parser = argparse.ArgumentParser(description="Generate golden_response field for router test cases")
    parser.add_argument("--in", dest="in_path", default=str(default_in),
                        help=f"input JSON path (default: {default_in})")
    parser.add_argument("--out", dest="out_path", default=None,
                        help="output JSON path (default: same as --in, in-place)")
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite existing golden_response values")
    args = parser.parse_args()

    in_path  = Path(args.in_path)
    out_path = Path(args.out_path) if args.out_path else in_path

    changed = populate(in_path, out_path, overwrite=args.overwrite)
    print(f"[generate_golden_responses] populated {changed} case(s) → {out_path}")


if __name__ == "__main__":
    main()
