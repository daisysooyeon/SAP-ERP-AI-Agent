"""
src/evaluation/eval_rag.py
RAG 품질 평가 — 자체 구현 메트릭 (RAGAS 대신)

평가 메트릭:
  Hit Rate      : top-k 안에 정답 source_chunk가 포함된 비율 (목표 ≥ 85%)
  NDCG@3        : 상위 3개 청크 순위 품질, 정답이 높을수록 높은 점수 (목표 ≥ 0.75)
  MRR           : Mean Reciprocal Rank, 첫 정답 문서 역순위 평균 (목표 ≥ 0.70)
  Context Recall: top-k 컨텍스트가 정답 생성에 필요한 정보를 포함하는지 (목표 ≥ 0.90)
  Faithfulness  : 생성 답변이 retrieved 컨텍스트에 근거하는지 LLM judge (목표 ≥ 0.85)

입력 데이터셋 형식 (rag_test_cases.json):
  {
    "id": "rag_0001",
    "question": "What is ...",
    "answer": "ground truth short answer",
    "source_chunk": "exact chunk text from the PDF",
    "source_doc": "TS460_2_EN_Col23.pdf",
    "chunk_id": 69,
    "question_type": "factual"
  }

CLI 사용법:
  python -m src.evaluation.eval_rag                              # 전체 평가
  python -m src.evaluation.eval_rag --skip-llm-judge             # retrieval 메트릭만 (빠름)
  python -m src.evaluation.eval_rag --report reports/rag_eval.json
  python -m src.evaluation.eval_rag --limit 10                   # 처음 10개만
"""

from __future__ import annotations

import logging
import math
import os
import time

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import get_config
from src.rag.retriever import build_hybrid_retriever
from src.rag.reranker import rerank
from src.graph.worker_b import _serialize_docs

load_dotenv()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM for judge (gemini-flash)
# ---------------------------------------------------------------------------

_judge_llm: ChatOpenAI | None = None


def _get_judge_llm() -> ChatOpenAI:
    global _judge_llm
    if _judge_llm is not None:
        return _judge_llm
    cfg = get_config()
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    judge_cfg = cfg.models.eval_judge
    _judge_llm = ChatOpenAI(
        model=judge_cfg.name,
        temperature=0.0,
        openai_api_key=api_key,
        openai_api_base=cfg.openrouter.base_url,
        default_headers={
            "HTTP-Referer": "https://github.com/daisysooyeon/SAP-ERP-AI-Agent",
            "X-Title": "SAP-ERP-AI-Agent RAG-Eval",
        },
    )
    return _judge_llm


# ---------------------------------------------------------------------------
# 매칭 헬퍼
# ---------------------------------------------------------------------------

def _chunk_in_docs(source_chunks: str | list[str], docs: list[dict], threshold: float = 0.5) -> int | None:
    """
    source_chunks(문자열 또는 리스트) 중 하나라도 retrieved docs 중 어느 위치(0-indexed)에 있는지 반환.
    정확한 substring이 없으면 50% 이상 토큰 겹침으로 판단.
    여러 개 매칭될 경우 가장 높은 순위(작은 인덱스) 반환.
    없으면 None 반환.
    """
    if not source_chunks:
        return None
        
    if isinstance(source_chunks, str):
        chunks_to_check = [source_chunks]
    else:
        chunks_to_check = source_chunks

    best_pos = None

    for chunk in chunks_to_check:
        gt_tokens = set(chunk.lower().split())
        for i, doc in enumerate(docs):
            content = doc["content"]
            # 1순위: substring
            if chunk[:80].strip() in content:
                if best_pos is None or i < best_pos:
                    best_pos = i
                break
            # 2순위: 토큰 overlap
            doc_tokens = set(content.lower().split())
            if gt_tokens and doc_tokens:
                overlap = len(gt_tokens & doc_tokens) / len(gt_tokens)
                if overlap >= threshold:
                    if best_pos is None or i < best_pos:
                        best_pos = i
                    break
                    
    return best_pos


# ---------------------------------------------------------------------------
# Retrieval 메트릭 (Hit Rate, NDCG@3, MRR)
# ---------------------------------------------------------------------------

def compute_hit_rate(cases: list[dict], all_retrieved: list[list[dict]]) -> float:
    """top-k 안에 정답 청크가 포함된 비율"""
    hits = sum(
        1 for case, docs in zip(cases, all_retrieved)
        if _chunk_in_docs(case.get("rag_evidence"), docs) is not None
    )
    return hits / len(cases) if cases else 0.0


def compute_ndcg_at_3(cases: list[dict], all_retrieved: list[list[dict]]) -> float:
    """
    NDCG@3: 정답이 top-3 안에서 몇 번째에 있는지에 따라 점수 차등
    IDCG = 1/log2(2) = 1.0 (정답이 1위인 이상적 케이스)
    """
    ndcg_scores = []
    for case, docs in zip(cases, all_retrieved):
        top3 = docs[:3]
        pos = _chunk_in_docs(case.get("rag_evidence"), top3)
        if pos is None:
            ndcg_scores.append(0.0)
        else:
            dcg = 1.0 / math.log2(pos + 2)   # pos는 0-indexed, log2(rank+1)
            idcg = 1.0 / math.log2(2)         # 이상적: rank=1
            ndcg_scores.append(dcg / idcg)
    return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0


def compute_mrr(cases: list[dict], all_retrieved: list[list[dict]]) -> float:
    """Mean Reciprocal Rank: 첫 번째 정답 문서의 역순위 평균"""
    rr_scores = []
    for case, docs in zip(cases, all_retrieved):
        pos = _chunk_in_docs(case.get("rag_evidence"), docs)
        if pos is None:
            rr_scores.append(0.0)
        else:
            rr_scores.append(1.0 / (pos + 1))   # 0-indexed → 1-indexed
    return sum(rr_scores) / len(rr_scores) if rr_scores else 0.0


# ---------------------------------------------------------------------------
# LLM Judge 메트릭 (Context Recall, Faithfulness)
# ---------------------------------------------------------------------------

_CONTEXT_RECALL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an evaluation assistant. "
        "Given a question and a context, "
        "determine whether the context contains enough information to answer the question correctly.\n"
        "Respond with ONLY '1' (yes, context is sufficient) or '0' (no, context is insufficient)."
    )),
    ("human", (
        "Question: {question}\n\n"
        "Context:\n{context}\n\n"
        "Does the context contain sufficient information? (1 or 0):"
    )),
])

_FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an evaluation assistant. "
        "Given an answer and the context it was generated from, "
        "determine whether the answer is fully supported by the context (no hallucination).\n"
        "Respond with ONLY '1' (yes, answer is faithful) or '0' (no, answer contains hallucination)."
    )),
    ("human", (
        "Context:\n{context}\n\n"
        "Answer: {rag_answer}\n\n"
        "Is the answer fully supported by the context? (1 or 0):"
    )),
])


# 무효 응답 카운터 — 메트릭별 분리 (0/1 이외의 응답 횟수)
_judge_invalid_context_recall: int = 0
_judge_invalid_faithfulness:   int = 0


def _llm_judge(
    prompt_template: ChatPromptTemplate,
    variables: dict,
    metric: str = "unknown",
    delay: float = 0.5,
) -> int:
    """LLM judge 호출 → 0 또는 1 반환. 0/1 이외 응답은 metric별 카운터에 집계."""
    global _judge_invalid_context_recall, _judge_invalid_faithfulness
    try:
        llm = _get_judge_llm()
        chain = prompt_template | llm
        result = chain.invoke(variables)
        # qwen3-8b는 <think>...</think> 블록 또는 줄바꿈 후 숫자를 출력하는 경우가 있음
        import re as _re
        raw_val = result.content.strip()
        # <think>...</think> 제거
        val = _re.sub(r"<think>.*?</think>", "", raw_val, flags=_re.DOTALL).strip()
        # 첫 번째 등장하는 0 또는 1 추출
        m = _re.search(r"[01]", val)
        val = m.group() if m else val
        time.sleep(delay)
        if val.startswith("1"):
            return 1
        elif val.startswith("0"):
            return 0
        else:
            if metric == "context_recall":
                _judge_invalid_context_recall += 1
            else:
                _judge_invalid_faithfulness += 1
            logger.warning(
                "[eval_rag] LLM judge [%s] unexpected response: %r",
                metric, val[:80],
            )
            return 0   # conservative: unexpected 응답은 0점 처리
    except Exception as e:
        logger.warning("[eval_rag] LLM judge [%s] failed: %s", metric, e)
        return 0


def compute_context_recall(
    cases: list[dict],
    all_retrieved: list[list[dict]],
    rag_answers: list[str],
) -> float:
    """Context Recall: retrieved context가 정답 생성에 충분한지 LLM judge (목표 >= 0.90)"""
    scores = []
    for case, docs in zip(cases, all_retrieved):
        context = "\n\n".join(d["content"] for d in docs)
        score = _llm_judge(
            _CONTEXT_RECALL_PROMPT,
            {"question": case.get("qa_question") or case.get("question"), "context": context},
            metric="context_recall",
        )
        scores.append(score)
        logger.debug("[eval_rag] Context Recall [%s]: %d", case.get("id"), score)
    return sum(scores) / len(scores) if scores else 0.0


def compute_faithfulness(
    cases: list[dict],
    all_retrieved: list[list[dict]],
    rag_answers: list[str],
) -> float:
    """Faithfulness: 생성 답변이 컨텍스트에 근거하는지 LLM judge (목표 >= 0.85)"""
    scores = []
    for case, docs, answer in zip(cases, all_retrieved, rag_answers):
        if not answer or not docs:
            scores.append(0)
            continue
        context = "\n\n".join(d["content"] for d in docs)
        score = _llm_judge(
            _FAITHFULNESS_PROMPT,
            {"context": context, "rag_answer": answer},
            metric="faithfulness",
        )
        scores.append(score)
        logger.debug("[eval_rag] Faithfulness [%s]: %d", case.get("id"), score)
    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# 메인 평가 함수
# ---------------------------------------------------------------------------

def run_rag_evaluation(
    test_cases: list[dict],
    rag_answers: list[str] | None = None,
    all_retrieved: list[list[dict]] | None = None,
    skip_llm_judge: bool = False,
) -> dict:
    """
    전체 RAG 평가 실행.

    Args:
        test_cases:    rag_test_cases.json 로드 결과
        rag_answers:   각 케이스에 대한 생성 답변 (없으면 retrieval 메트릭만 계산)
        all_retrieved: 각 케이스의 retrieved docs 목록
        skip_llm_judge: True이면 Context Recall / Faithfulness 스킵

    Returns:
        {
            "hit_rate": float,
            "ndcg_at_3": float,
            "mrr": float,
            "context_recall": float | None,
            "faithfulness": float | None,
            "n_cases": int,
        }
    """
    n = len(test_cases)
    logger.info("[eval_rag] Evaluating %d cases ...", n)

    # LLM judge 실행 전 카운터 리셋
    global _judge_invalid_context_recall, _judge_invalid_faithfulness
    _judge_invalid_context_recall = 0
    _judge_invalid_faithfulness   = 0

    hit_rate   = compute_hit_rate(test_cases, all_retrieved)
    ndcg_at_3  = compute_ndcg_at_3(test_cases, all_retrieved)
    mrr        = compute_mrr(test_cases, all_retrieved)

    context_recall: float | None = None
    faithfulness:   float | None = None

    if not skip_llm_judge and rag_answers:
        logger.info("[eval_rag] Running LLM judges (Context Recall + Faithfulness) ...")
        context_recall = compute_context_recall(test_cases, all_retrieved, rag_answers)
        faithfulness   = compute_faithfulness(test_cases, all_retrieved, rag_answers)

    if _judge_invalid_context_recall > 0:
        logger.warning(
            "[eval_rag] Context Recall judge invalid: %d / %d",
            _judge_invalid_context_recall, n,
        )
    if _judge_invalid_faithfulness > 0:
        logger.warning(
            "[eval_rag] Faithfulness judge invalid: %d / %d",
            _judge_invalid_faithfulness, n,
        )

    return {
        "hit_rate":                        round(hit_rate, 4),
        "ndcg_at_3":                       round(ndcg_at_3, 4),
        "mrr":                             round(mrr, 4),
        "context_recall":                  round(context_recall, 4) if context_recall is not None else None,
        "faithfulness":                    round(faithfulness, 4)   if faithfulness   is not None else None,
        "n_cases":                         n,
        "judge_invalid_context_recall":    _judge_invalid_context_recall,
        "judge_invalid_faithfulness":      _judge_invalid_faithfulness,
    }


# ---------------------------------------------------------------------------
# CLI 진입점 (python -m src.evaluation.eval_rag 또는 run_eval_rag.sh 에서 호출)
# ---------------------------------------------------------------------------

TARGETS = {
    "hit_rate":       0.85,
    "ndcg_at_3":      0.75,
    "mrr":            0.70,
    "context_recall": 0.90,
    "faithfulness":   0.85,
}

METRIC_LABELS = {
    "hit_rate":       "Hit Rate    ",
    "ndcg_at_3":      "NDCG@3      ",
    "mrr":            "MRR         ",
    "context_recall": "Ctx Recall  ",
    "faithfulness":   "Faithfulness",
}


def _main():
    import argparse
    import json
    import sys
    from pathlib import Path

    # Windows cp949 터미널 인코딩 대응
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from src.logging_config import setup_logging
    setup_logging()

    parser = argparse.ArgumentParser(description="RAG Quality Evaluation")
    parser.add_argument("--report",  default="reports/worker_b_eval.json", help="출력 리포트 경로")
    parser.add_argument("--dataset", default="data/eval/router_test_cases_gen.json")
    parser.add_argument("--skip-llm-judge", action="store_true", help="LLM judge 메트릭 스킵")
    args = parser.parse_args()

    # 데이터셋 로드 및 필터링 (QA_ONLY, BOTH)
    dataset_path = Path(args.dataset)
    all_cases: list[dict] = json.loads(dataset_path.read_text(encoding="utf-8"))
    test_cases = [c for c in all_cases if c.get("label") in ("QA_ONLY", "BOTH")]
    logger.info("Loaded %d test cases for Worker B from %s", len(test_cases), dataset_path)

    # 리트리버 / 리랭커 초기화
    logger.info("Initializing retriever (bge-m3 loading may take a moment) ...")
    retriever = build_hybrid_retriever()
    cfg = get_config()
    top_n = cfg.rag.top_k_rerank

    # 각 케이스 실행
    results = []
    all_retrieved: list[list[dict]] = []
    rag_answers:   list[str]        = []

    print(f"\n{'='*60}")
    print(f"  RAG Evaluation - {len(test_cases)} cases")
    print(f"{'='*60}\n")

    for i, case in enumerate(test_cases):
        cid      = case.get("id", f"case_{i}")
        # router_test_cases_gen.json의 input (이메일 원문) 사용
        raw_input = case.get("input") or case.get("user_input") or case.get("qa_question") or case.get("question", "")
        t0       = time.perf_counter()

        # 쿼리 추출 — Query Expansion (worker_b의 실제 로직과 동일하게)
        from src.graph.worker_b import _extract_rag_queries
        rag_queries = _extract_rag_queries(raw_input)
        if not rag_queries:
            rag_queries = [raw_input]  # fallback
        rag_query = rag_queries[0]  # 대표 쿼리 (reranking 기준)

        # 검색 — 쿼리별 검색 후 중복 제거 병합
        try:
            seen_contents: set[str] = set()
            candidate_docs = []
            for q in rag_queries:
                for doc in retriever.invoke(q):
                    if doc.page_content not in seen_contents:
                        seen_contents.add(doc.page_content)
                        candidate_docs.append(doc)
        except Exception as e:
            logger.error("[%s] Retrieval failed: %s", cid, e)
            candidate_docs = []

        # 리랭킹 (multi-query)
        try:
            top_docs = rerank(rag_query, candidate_docs, top_n=top_n, queries=rag_queries)
        except Exception as e:
            logger.warning("[%s] Rerank failed: %s. Using top-k.", cid, e)
            top_docs = candidate_docs[:top_n]

        # 직렬화
        docs_serial = _serialize_docs(top_docs)

        # 답변 생성 (LLM judge 필요 시)
        answer = ""
        if not args.skip_llm_judge:
            from src.graph.worker_b import _generate_answer
            answer = _generate_answer(rag_query, top_docs)

        elapsed = (time.perf_counter() - t0) * 1000

        # 정답 위치
        hit_pos = _chunk_in_docs(case.get("rag_evidence"), docs_serial)

        result = {
            "id":             cid,
            "question":       rag_query,
            "raw_input":      raw_input[:100],
            "label":          case.get("label", ""),
            "hit":            hit_pos is not None,
            "hit_position":   hit_pos,
            "rag_answer":     answer,
            "retrieved_docs": docs_serial,
            "elapsed_ms":     round(elapsed, 1),
        }
        results.append(result)
        all_retrieved.append(docs_serial)
        rag_answers.append(answer)

        status = f"HIT@{hit_pos+1}" if hit_pos is not None else "MISS"
        print(f"  [{cid}] {status} | {elapsed:.0f}ms | {rag_query[:60]!r}")

    # 집계
    print(f"\n{'='*60}")
    print("  Computing metrics ...")

    summary = run_rag_evaluation(
        test_cases=test_cases,
        rag_answers=rag_answers if not args.skip_llm_judge else None,
        all_retrieved=all_retrieved,
        skip_llm_judge=args.skip_llm_judge,
    )

    # 결과 출력
    print(f"\n{'='*60}")
    print("  RAG Evaluation Summary")
    print(f"{'='*60}")
    for key, label in METRIC_LABELS.items():
        val = summary.get(key)
        if val is None:
            print(f"  {label}: (skipped)")
            continue
        target = TARGETS.get(key, 0)
        pct    = f"{val*100:.1f}%"
        flag   = "OK" if val >= target else "FAIL"
        print(f"  {label}: {pct:>7}  (target: {target*100:.0f}%)  [{flag}]")
    print(f"  {'Cases':12}: {summary['n_cases']}")
    if not args.skip_llm_judge:
        inv_cr = summary.get("judge_invalid_context_recall", 0)
        inv_fa = summary.get("judge_invalid_faithfulness", 0)
        print(f"  {'Ctx Invalid':12}: {inv_cr:>3}  [{'OK' if inv_cr == 0 else 'WARN'}]")
        print(f"  {'Faith Invalid':12}: {inv_fa:>3}  [{'OK' if inv_fa == 0 else 'WARN'}]")
    print(f"{'='*60}\n")

    # 리포트 저장
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"summary": summary, "targets": TARGETS, "results": results}
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report saved to: {report_path}\n")


if __name__ == "__main__":
    _main()
