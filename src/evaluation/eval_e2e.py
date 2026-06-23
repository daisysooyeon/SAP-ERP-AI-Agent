"""
src/evaluation/eval_e2e.py
End-to-End evaluation — LLM-as-a-Judge

Runs the full agent pipeline on each test case and scores the final email
response on three metrics using a Judge LLM:

  - faithfulness  (1-5): No hallucinated facts outside the provided context
  - correctness   (1-5): ERP/RAG results accurately reflected in the reply
  - format        (1-5): Proper business email structure (greeting, body, sign-off)

Usage:
  python -m src.evaluation.eval_e2e
  python -m src.evaluation.eval_e2e --dry-run
  python -m src.evaluation.eval_e2e --report reports/e2e_eval_result.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from src.config import get_config

from dotenv import load_dotenv

from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.graph.state import AgentState
from src.graph_builder import _build_state_graph
from src.evaluation._bertscore import compute_bertscore_f1


load_dotenv()

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    id:              str
    label:           str
    user_input:      str          # first 120 chars (preview)

    # Agent outputs
    intent:          Optional[str]  = None
    erp_status:      Optional[str]  = None
    erp_action:      Optional[dict] = None
    errors:          list           = field(default_factory=list)
    final_response:  Optional[str]  = None
    rag_answer:      Optional[str]  = None
    rag_evidence:    Optional[str]  = None   # ground-truth RAG evidence from test case
    golden_response: Optional[str]  = None   # ground-truth final email reply (Samsung SDS 형식 템플릿)

    # 구조화 정답 대비 결정론적 exact-match (LLM judge 없이 정량 채점).
    # 데이터셋의 expected_* 필드를 정답으로 사용 — action/BOTH 케이스에만 적용(QA_ONLY는 None).
    expected_action_type:        Optional[str]  = None
    expected_erp_action_status:  Optional[str]  = None
    action_type_match:           Optional[bool] = None  # None=비적용(정답 없음)
    erp_status_match:            Optional[bool] = None

    # Judge scores (1-5, or None if judging failed)
    faithfulness:    Optional[float] = None
    correctness:     Optional[float] = None
    format_score:    Optional[float] = None
    overall:         Optional[float] = None
    judge_reasoning: Optional[str]   = None
    judge_error:     Optional[str]   = None

    # Samsung SDS 물류팀 형식 정적 검사 (LLM judge의 format_score와 별개,
    # 결정론적이라 회귀 추적용으로 더 신뢰 가능)
    format_checks:     dict = field(default_factory=dict)  # {rule: bool, ...}
    format_rule_score: int  = 0  # 통과 규칙 수
    format_rule_total: int  = 0  # 전체 규칙 수
    format_compliant:  bool = False  # 모든 규칙 통과 여부 (score == total)

    # BERTScore F1 — golden_response 대비 의미 유사도 (0.0~1.0).
    # None은 비교 불가(빈 정답/빈 생성/모델 로드 실패) — 집계에서 자동 제외.
    bertscore_f1:    Optional[float] = None

    elapsed_ms:      int = 0


@dataclass
class EvalReport:
    generated_at:      str
    model_synthesizer: str
    model_judge:       str
    total:             int = 0

    # Aggregate averages
    avg_faithfulness:  float = 0.0
    avg_correctness:   float = 0.0
    avg_format:        float = 0.0
    avg_overall:       float = 0.0

    # Samsung SDS 형식 정적 검사 집계
    format_compliant_total:  int = 0   # 6/6 모두 통과한 케이스 수
    format_compliance_rate:  float = 0.0  # format_compliant_total / total
    format_rule_pass:        dict = field(default_factory=dict)  # {rule_name: 통과 케이스 수}
    format_avg_score:        float = 0.0  # 케이스당 평균 통과율 (score/total)

    # BERTScore 집계 — 정답지 대비 평균 의미 유사도 + 유효 케이스 수
    avg_bertscore_f1:     Optional[float] = None
    bertscore_valid_n:    int = 0

    # 구조화 정답 exact-match accuracy (LLM judge 없는 결정론적 정량 채점).
    # 분모 = expected_*가 명시된 케이스 수(action/BOTH). 분자 = 정확히 일치한 수.
    action_type_accuracy:   Optional[float] = None
    action_type_match_n:    int = 0
    action_type_total:      int = 0
    erp_status_accuracy:    Optional[float] = None
    erp_status_match_n:     int = 0
    erp_status_total:       int = 0

    results: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Samsung SDS 물류팀 형식 정적 검사
# ---------------------------------------------------------------------------
#
# synthesizer는 다음 6개 라인이 반드시 들어가는 응답을 만들어야 한다. LLM judge의
# format_score는 1~5점이라 회귀가 잡히기 어려우니, 결정론적 정규식으로 같이 본다.
# (각 패턴은 너무 엄격하지 않게 — 대소문자 + 가벼운 공백 변형 허용)
#
#   r1 greeting_dear         : "Dear ..." 인사
#   r2 hello_line            : "Hello," 또는 "Hello." 한 줄
#   r3 samsung_sds_identify  : "This is the Samsung SDS Logistics Team."
#   r4 internal_review_intro : "After our internal review of the matter you raised in your email,"
#   r5 contact_invitation    : "Please feel free to contact us anytime for any related inquiries."
#   r6 thank_you_signoff     : "Thank you." 종결
#
# 추가 안티-규칙으로 옛 sign-off가 남았는지도 본다:
#   r7 no_legacy_signoff     : "Best regards, SAP ERP Support Team"이 없어야 통과
#                              (옛 템플릿이 새 템플릿과 섞이는 회귀 잡기)

_FORMAT_PATTERNS = {
    "greeting_dear":         r"^\s*Dear\s+[^\n,]+,",
    "hello_line":            r"(?m)^\s*Hello[,.]\s*$",
    "samsung_sds_identify":  r"This\s+is\s+the\s+Samsung\s+SDS\s+Logistics\s+Team\.",
    "internal_review_intro": r"After\s+our\s+internal\s+review\s+of\s+the\s+matter\s+you\s+raised\s+in\s+your\s+email,",
    "contact_invitation":    r"Please\s+feel\s+free\s+to\s+contact\s+us\s+anytime\s+for\s+any\s+related\s+inquiries\.",
    "thank_you_signoff":     r"(?m)^\s*Thank\s+you\.\s*$",
}

# 옛 sign-off가 남으면 fail로 잡는 안티-패턴 (regex 매칭 = 위반)
_FORMAT_ANTI_PATTERNS = {
    "no_legacy_signoff": r"Best\s+regards,\s*\n?\s*SAP\s+ERP\s+Support\s+Team",
}


def _check_samsung_sds_format(reply: str | None) -> tuple[dict, int, int]:
    """final_response가 Samsung SDS 물류팀 형식을 지켰는지 정적 검사.

    Returns:
        (checks, score, total) — checks는 {rule_name: bool}, score=통과 규칙 수
    """
    text = reply or ""
    checks: dict[str, bool] = {
        name: bool(re.search(pat, text, re.IGNORECASE))
        for name, pat in _FORMAT_PATTERNS.items()
    }
    for name, pat in _FORMAT_ANTI_PATTERNS.items():
        # anti-pattern은 매칭되지 않아야 통과
        checks[name] = not bool(re.search(pat, text, re.IGNORECASE))

    score = sum(checks.values())
    total = len(checks)
    return checks, score, total


# ---------------------------------------------------------------------------
def _build_eval_graph():
    """HITL 없이 그래프 빌드 — 평가 중 interrupt 없이 worker_a → synthesizer 직통."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    memory = SqliteSaver(conn)
    return _build_state_graph(hitl=False).compile(checkpointer=memory)


# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """\
You are an expert evaluator of AI-generated customer service email responses for SAP ERP systems.

Score the AI email reply on three dimensions, each from 1 to 5:

1. faithfulness  (1-5): Does the reply avoid making up facts not present in the provided context
   or the ground-truth RAG evidence? Compare the reply against both the RAG Answer and the
   Ground-Truth Evidence to detect hallucinations.
   5 = zero hallucination, 1 = heavily fabricated content

2. correctness   (1-5): Does the reply accurately reflect the ERP processing result and answer
   the customer's question? Use the Ground-Truth Golden Reply (if provided) as the reference
   for both content accuracy and the points that should be covered.
   5 = matches the golden reply in substance, 1 = contradicts it

3. format        (1-5): Does the reply follow the required Samsung SDS Logistics Team email
   format? The reply MUST:
     - open with "Dear {{name}}," (use first name if available, else "Dear Customer,")
     - have a "Hello," line
     - identify the sender as "This is the Samsung SDS Logistics Team."
     - introduce the body with "After our internal review of the matter you raised in your email,"
     - close with "Please feel free to contact us anytime for any related inquiries."
     - end with "Thank you."
   5 = follows every structural line exactly; 3 = follows most but a line is missing/altered;
   1 = entirely different structure (e.g. "Best regards, SAP ERP Support Team")

Return ONLY a valid JSON object, no extra text:
{{"faithfulness": <int>, "correctness": <int>, "format": <int>, "reasoning": "<one sentence>"}}
"""

_JUDGE_HUMAN = """\
=== INPUT ===
User email: {user_input}

=== CONTEXT PROVIDED TO THE AI ===
Intent      : {intent}
ERP Status  : {erp_status}
ERP Action  : {erp_action_summary}
RAG Answer  : {rag_answer}
Errors      : {errors}

=== GROUND-TRUTH RAG EVIDENCE ===
{rag_evidence}

=== GROUND-TRUTH GOLDEN REPLY (reference for correctness + format) ===
{golden_response}

=== AI REPLY TO EVALUATE ===
{final_response}
"""


def _get_judge_llm():
    from langchain_openai import ChatOpenAI
    from src.config import get_config

    cfg     = get_config()
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set.")

    j_cfg = cfg.models.eval_judge
    return ChatOpenAI(
        model=j_cfg.name,
        temperature=j_cfg.temperature,
        openai_api_key=api_key,
        openai_api_base=cfg.openrouter.base_url,
        default_headers={
            "HTTP-Referer": "https://github.com/daisysooyeon/SAP-ERP-AI-Agent",
            "X-Title":      "SAP-ERP-AI-Agent EvalJudge",
        },
    )


def _judge(cr: CaseResult) -> None:
    """Call Judge LLM and populate score fields on cr in-place."""

    if not cr.final_response:
        cr.judge_error = "No final_response to judge"
        return

    prompt = ChatPromptTemplate.from_messages([
        ("system", _JUDGE_SYSTEM),
        ("human",  _JUDGE_HUMAN),
    ])

    try:
        llm    = _get_judge_llm()
        chain  = prompt | llm | JsonOutputParser()
        from src.graph.synthesizer import _erp_action_summary
        result = chain.invoke({
            "user_input":         cr.user_input,
            "intent":             cr.intent or "UNKNOWN",
            "erp_status":         cr.erp_status or "N/A",
            "erp_action_summary": _erp_action_summary(cr.erp_action or {}, cr.erp_status),
            "rag_answer":         cr.rag_answer or "None",
            "errors":             ", ".join(cr.errors) if cr.errors else "None",
            "rag_evidence":       cr.rag_evidence or "None",
            "golden_response":    cr.golden_response or "None",
            "final_response":     cr.final_response,
        })

        cr.faithfulness    = float(result.get("faithfulness", 0))
        cr.correctness     = float(result.get("correctness", 0))
        cr.format_score    = float(result.get("format", 0))
        cr.overall         = round(
            (cr.faithfulness + cr.correctness + cr.format_score) / 3, 2
        )
        cr.judge_reasoning = result.get("reasoning", "")

    except Exception as e:
        cr.judge_error = str(e)
        logger.error("[e2e] Judge failed for %s: %s", cr.id, e)


# ---------------------------------------------------------------------------
# Per-case runner
# ---------------------------------------------------------------------------

def _run_case(tc: dict, graph) -> CaseResult:
    t0        = time.perf_counter()
    thread_id = str(uuid.uuid4())
    config    = {"configurable": {"thread_id": thread_id}}

    cr = CaseResult(
        id=tc.get("id", ""),
        label=tc.get("label", ""),
        user_input=tc.get("user_input", "")[:120],
        rag_evidence=tc.get("rag_evidence"),
        golden_response=tc.get("golden_response"),
        expected_action_type=tc.get("expected_action_type"),
        expected_erp_action_status=tc.get("expected_erp_action_status"),
    )

    # ── Run agent ─────────────────────────────────────────────────────────────
    try:
        # LangGraph 진입 전 이메일 전처리 — 운영 경로와 동일하게 e2e에서도 적용
        from src.preprocess import preprocess_email
        email_text = tc.get("user_input", "")
        email_ctx  = preprocess_email(email_text)

        result = graph.invoke(
            {
                "user_input":              email_text,
                "email_context":           email_ctx.model_dump(),
                "error_messages":          [],
                "requires_human_approval": False,
                "human_approved":          None,
            },
            config=config,
        )
        cr.intent         = result.get("intent")
        cr.erp_status     = result.get("erp_action_status")
        cr.erp_action     = result.get("erp_action") or {}
        cr.errors         = result.get("error_messages") or []
        cr.final_response = result.get("final_response")
        cr.rag_answer     = result.get("rag_answer") or ""

        logger.info(
            "[e2e] [%s] agent done — intent=%s erp_status=%s response_len=%d",
            cr.id, cr.intent, cr.erp_status,
            len(cr.final_response or ""),
        )
    except Exception as e:
        logger.error("[e2e] [%s] Agent run failed: %s", cr.id, e)
        cr.judge_error = f"Agent run failed: {e}"
        cr.elapsed_ms  = int((time.perf_counter() - t0) * 1000)
        return cr

    # ── 구조화 정답 exact-match (결정론적, LLM judge 없이 정량 채점) ───────────
    # expected_*가 명시된 경우(action/BOTH)에만 비교. QA_ONLY는 None 유지(비적용).
    if cr.expected_action_type is not None:
        got_at = (cr.erp_action or {}).get("action_type")
        cr.action_type_match = (got_at == cr.expected_action_type)
    if cr.expected_erp_action_status is not None:
        cr.erp_status_match = (cr.erp_status == cr.expected_erp_action_status)

    # ── Samsung SDS 형식 정적 검사 ────────────────────────────────────────────
    # LLM judge보다 먼저, 결정론적으로 수행. judge가 실패해도 형식 회귀는 잡힌다.
    fmt_checks, fmt_score, fmt_total = _check_samsung_sds_format(cr.final_response)
    cr.format_checks     = fmt_checks
    cr.format_rule_score = fmt_score
    cr.format_rule_total = fmt_total
    cr.format_compliant  = (fmt_score == fmt_total)
    if not cr.format_compliant:
        missing = [name for name, ok in fmt_checks.items() if not ok]
        logger.warning("[e2e] [%s] format non-compliant — missing/violated: %s",
                       cr.id, missing)

    # ── Judge ─────────────────────────────────────────────────────────────────
    _judge(cr)

    cr.elapsed_ms = int((time.perf_counter() - t0) * 1000)
    score_str = (
        f"F={cr.faithfulness} C={cr.correctness} Fmt={cr.format_score} "
        f"Overall={cr.overall}"
        if cr.faithfulness is not None
        else f"judge_error={cr.judge_error}"
    )
    logger.info("[e2e] [%s] %s | fmt_rules=%d/%d | %dms",
                cr.id, score_str, cr.format_rule_score, cr.format_rule_total, cr.elapsed_ms)
    return cr


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------

def run_eval(
    test_cases_path: str,
    report_path: Optional[str],
    dry_run: bool,
    resume_path: Optional[str] = None,
) -> None:

    cfg = get_config()

    all_cases: list[dict] = json.loads(
        Path(test_cases_path).read_text(encoding="utf-8")
    )
    logger.info("Loaded %d E2E test cases", len(all_cases))

    # ── Resume mode: re-run only failed cases from an existing report ─────────
    existing_results: dict[str, dict] = {}
    if resume_path:
        existing_report = json.loads(
            Path(resume_path).read_text(encoding="utf-8")
        )
        existing_results = {r["id"]: r for r in existing_report["results"]}
        failed_ids = {
            r["id"] for r in existing_report["results"]
            if r.get("faithfulness") is None
        }
        all_cases = [tc for tc in all_cases if tc["id"] in failed_ids]
        logger.info(
            "Resume mode: %d failed cases to re-run (out of %d total)",
            len(all_cases), len(existing_results),
        )

    if dry_run:
        all_cases = all_cases[:3]
        logger.info("Dry-run mode: using first %d cases only", len(all_cases))

    report = EvalReport(
        generated_at      = datetime.now(timezone.utc).isoformat(),
        model_synthesizer = cfg.models.synthesizer.name,
        model_judge       = cfg.models.eval_judge.name,
        total             = len(all_cases),
    )

    graph = _build_eval_graph()

    for tc in all_cases:
        cr = _run_case(tc, graph)
        if resume_path:
            existing_results[cr.id] = asdict(cr)
        else:
            report.results.append(asdict(cr))

    # ── Merge resume results back into full report ────────────────────────────
    if resume_path:
        report.results = list(existing_results.values())
        report.total   = len(report.results)

    # ── Aggregate scores ──────────────────────────────────────────────────────
    scored = [
        r for r in report.results
        if r.get("faithfulness") is not None
    ]

    if scored:
        report.avg_faithfulness = round(
            sum(r["faithfulness"] for r in scored) / len(scored), 2
        )
        report.avg_correctness  = round(
            sum(r["correctness"]  for r in scored) / len(scored), 2
        )
        report.avg_format       = round(
            sum(r["format_score"] for r in scored) / len(scored), 2
        )
        report.avg_overall      = round(
            sum(r["overall"]      for r in scored) / len(scored), 2
        )

    # ── BERTScore F1 ─────────────────────────────────────────────────────────
    # golden_response 대비 final_response의 의미 유사도. LLM judge의 correctness가
    # 1~5 정성 평가라면, BERTScore는 0~1 정량 의미 유사도. 모델 로드/배치 호출
    # 비용 때문에 케이스별로 호출하지 않고 여기서 한 번에 처리한다.
    bs_candidates = [r.get("final_response") or "" for r in report.results]
    bs_references = [r.get("golden_response") or "" for r in report.results]
    per_case_bs, mean_bs = compute_bertscore_f1(bs_candidates, bs_references)
    for r, val in zip(report.results, per_case_bs):
        r["bertscore_f1"] = val
    report.avg_bertscore_f1   = mean_bs
    report.bertscore_valid_n  = sum(1 for v in per_case_bs if v is not None)

    # ── 구조화 정답 exact-match accuracy 집계 (결정론적 정량 채점) ──────────────
    at_results = [r for r in report.results if r.get("action_type_match") is not None]
    st_results = [r for r in report.results if r.get("erp_status_match") is not None]
    report.action_type_total   = len(at_results)
    report.action_type_match_n = sum(1 for r in at_results if r.get("action_type_match"))
    report.erp_status_total    = len(st_results)
    report.erp_status_match_n  = sum(1 for r in st_results if r.get("erp_status_match"))
    if report.action_type_total:
        report.action_type_accuracy = round(report.action_type_match_n / report.action_type_total, 4)
    if report.erp_status_total:
        report.erp_status_accuracy = round(report.erp_status_match_n / report.erp_status_total, 4)

    # Samsung SDS 형식 정적 검사는 judge 실패와 무관하게 final_response가 있으면
    # 모두 채워지므로 results 전체로 집계 (scored가 아님).
    fmt_rule_names  = list(_FORMAT_PATTERNS.keys()) + list(_FORMAT_ANTI_PATTERNS.keys())
    rule_counter    = {name: 0 for name in fmt_rule_names}
    score_sum       = 0
    fmt_total_rules = len(fmt_rule_names)
    for r in report.results:
        checks = r.get("format_checks") or {}
        for name, ok in checks.items():
            if ok:
                rule_counter[name] = rule_counter.get(name, 0) + 1
        score_sum += r.get("format_rule_score", 0)
        if r.get("format_compliant"):
            report.format_compliant_total += 1
    n = report.total
    if n:
        report.format_compliance_rate = round(report.format_compliant_total / n, 4)
        report.format_avg_score       = round(score_sum / (n * fmt_total_rules), 4)
    report.format_rule_pass = rule_counter

    # ── Summary printout ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  E2E Evaluation Results  (LLM-as-a-Judge)")
    print("=" * 60)
    print(f"  Synthesizer model : {report.model_synthesizer}")
    print(f"  Judge model       : {report.model_judge}")
    print(f"  Total cases       : {n}")
    print(f"  Scored cases      : {len(scored)}")
    print("-" * 60)
    print(f"  Faithfulness      : {report.avg_faithfulness:.1f} / 5.0")
    print(f"  Correctness       : {report.avg_correctness:.1f}  / 5.0")
    print(f"  Format (LLM judge): {report.avg_format:.1f}       / 5.0")
    print(f"  Overall           : {report.avg_overall:.1f}       / 5.0")
    print("-" * 60)
    bs_str = (
        f"{report.avg_bertscore_f1:.4f}"
        if report.avg_bertscore_f1 is not None else "n/a (bert_score 미설치/실패)"
    )
    print(f"  BERTScore F1      : {bs_str}  "
          f"({report.bertscore_valid_n}/{n} valid, golden 대비 의미 유사도, 0.0~1.0)")
    print("-" * 60)
    print("  Structured exact-match accuracy (deterministic, no LLM judge):")
    at_str = (f"{report.action_type_accuracy*100:.1f}%" if report.action_type_accuracy is not None else "n/a")
    st_str = (f"{report.erp_status_accuracy*100:.1f}%"  if report.erp_status_accuracy  is not None else "n/a")
    print(f"    action_type     : {at_str}  ({report.action_type_match_n}/{report.action_type_total} match, vs expected_action_type)")
    print(f"    erp_status      : {st_str}  ({report.erp_status_match_n}/{report.erp_status_total} match, vs expected_erp_action_status)")
    print("-" * 60)
    print("  Samsung SDS 형식 정적 검사 (결정론적):")
    print(f"    Fully compliant : {report.format_compliant_total}/{n}  "
          f"({report.format_compliance_rate*100:.1f}%)")
    print(f"    Avg rule pass   : {report.format_avg_score*100:.1f}%  "
          f"(per-case score / {fmt_total_rules} rules)")
    print("    Rule-level pass rate:")
    for rule_name, cnt in report.format_rule_pass.items():
        pct = (cnt / n * 100) if n else 0.0
        print(f"      {rule_name:<24}: {cnt}/{n}  ({pct:.1f}%)")
    print("=" * 60)

    # 형식 위반 케이스 상세 출력 (회귀 추적용)
    fmt_violations = [
        r for r in report.results
        if not r.get("format_compliant", True) and r.get("final_response")
    ]
    if fmt_violations:
        print(f"\n  Format violations ({len(fmt_violations)}):")
        for r in fmt_violations[:10]:  # 상위 10건만
            missing = [name for name, ok in (r.get("format_checks") or {}).items() if not ok]
            print(f"    [{r['id']}] missing/violated: {missing}")
        if len(fmt_violations) > 10:
            print(f"    … and {len(fmt_violations) - 10} more (see JSON report)")

    # Per-case breakdown
    print("\n  Per-case scores:")
    for r in report.results:
        if r.get("faithfulness") is not None:
            print(
                f"  [{r['id']}] ({r['label']:<12}) "
                f"F={r['faithfulness']}  C={r['correctness']}  "
                f"Fmt={r['format_score']}  → {r['overall']}"
            )
        else:
            print(f"  [{r['id']}] ({r['label']:<12}) ❌ {r.get('judge_error', 'error')}")
    print()

    # ── Save report ───────────────────────────────────────────────────────────
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Report saved → %s", report_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="E2E evaluation — LLM-as-a-Judge"
    )
    parser.add_argument(
        "--test-cases",
        default=str(_ROOT / "data" / "eval" / "router_test_cases_gen.json"),
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Path to save JSON report (omit to skip saving)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run only the first 3 cases",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="REPORT_JSON",
        help="Path to existing report JSON; re-runs only failed cases and merges results",
    )
    args = parser.parse_args()

    run_eval(
        test_cases_path=args.test_cases,
        report_path=args.report or args.resume,
        dry_run=args.dry_run,
        resume_path=args.resume,
    )
