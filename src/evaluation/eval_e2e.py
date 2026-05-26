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
  python -m src.evaluation.eval_e2e --report reports/e2e_eval.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
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
from langgraph.graph import END, StateGraph
from langgraph.types import Send
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.graph.router import router_node
from src.graph.state import AgentState
from src.graph.synthesizer import synthesizer_node
from src.graph.worker_a import worker_a_node
from src.graph.worker_b import worker_b_node


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
    rag_evidence:    Optional[str]  = None   # ground-truth from test case

    # Judge scores (1-5, or None if judging failed)
    faithfulness:    Optional[float] = None
    correctness:     Optional[float] = None
    format_score:    Optional[float] = None
    overall:         Optional[float] = None
    judge_reasoning: Optional[str]   = None
    judge_error:     Optional[str]   = None

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

    results: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Eval graph (HITL disabled — bypasses human_loop interrupt)
# ---------------------------------------------------------------------------

def _build_eval_graph():
    """
    Build a LangGraph graph without the human_loop node.
    ACTION_ONLY / BOTH emails go directly to synthesizer after worker_a.
    Uses an in-memory SQLite checkpointer (no file side-effects between runs).
    """

    builder = StateGraph(AgentState)
    builder.add_node("router",      router_node)
    builder.add_node("worker_a",    worker_a_node)
    builder.add_node("worker_b",    worker_b_node)
    builder.add_node("synthesizer", synthesizer_node)

    builder.set_entry_point("router")

    def route_after_router(state: AgentState):
        intent = state["intent"]
        if intent == "ACTION_ONLY":
            return [Send("worker_a", state)]
        elif intent == "QA_ONLY":
            return [Send("worker_b", state)]
        else:
            return [Send("worker_a", state), Send("worker_b", state)]

    builder.add_conditional_edges(
        "router", route_after_router, ["worker_a", "worker_b"]
    )
    # No HITL: worker_a goes straight to synthesizer
    builder.add_edge("worker_a",    "synthesizer")
    builder.add_edge("worker_b",    "synthesizer")
    builder.add_edge("synthesizer", END)

    conn   = sqlite3.connect(":memory:", check_same_thread=False)
    memory = SqliteSaver(conn)
    return builder.compile(checkpointer=memory)


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

2. correctness   (1-5): Does the reply accurately reflect the ERP processing result?
   For QA questions, does the reply align with the Ground-Truth Evidence?
   5 = perfectly aligned with the provided context, 1 = contradicts the context

3. format        (1-5): Does the reply follow proper business email structure?
   5 = perfect (greeting, clear body, professional sign-off), 1 = missing key structural elements

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
    )

    # ── Run agent ─────────────────────────────────────────────────────────────
    try:
        result = graph.invoke(
            {
                "user_input":              tc.get("user_input", ""),
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

    # ── Judge ─────────────────────────────────────────────────────────────────
    _judge(cr)

    cr.elapsed_ms = int((time.perf_counter() - t0) * 1000)
    score_str = (
        f"F={cr.faithfulness} C={cr.correctness} Fmt={cr.format_score} "
        f"Overall={cr.overall}"
        if cr.faithfulness is not None
        else f"judge_error={cr.judge_error}"
    )
    logger.info("[e2e] [%s] %s | %dms", cr.id, score_str, cr.elapsed_ms)
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

    n = report.total

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
    print(f"  Format            : {report.avg_format:.1f}       / 5.0")
    print(f"  Overall           : {report.avg_overall:.1f}       / 5.0")
    print("=" * 60)

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
