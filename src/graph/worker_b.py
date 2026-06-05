"""
src/graph/worker_b.py
Worker B: RAG 검색 + 답변 생성 노드

처리 흐름:
  ① 쿼리 추출  (LLM) — 이메일에서 핵심 SAP 질문을 3개 표현으로 확장 (Query Expansion)
  ② Hybrid Retrieval (Dense 0.5 + BM25 0.5) — 쿼리별 top-k 검색 후 중복 제거 병합
  ③ Reranking (bge-reranker-v2-m3) — top-5 재순위화
  ④ 답변 생성  (LLM) — top-5 컨텍스트 기반 RAG 답변

설정 값은 모두 configs.yaml에서 읽음:
  models.worker_b.name          = google/gemini-2.0-flash-lite-001
  rag.dense_weight              = 0.5
  rag.sparse_weight             = 0.5
  rag.top_k_retrieval           = 15
  rag.top_k_rerank              = 5
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import get_config
from src.graph.state import AgentState
from src.rag.retriever import build_hybrid_retriever
from src.rag.reranker import rerank

load_dotenv()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM builder (싱글턴)
# ---------------------------------------------------------------------------

_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is not None:
        return _llm

    cfg = get_config()
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set.")

    wb_cfg = cfg.models.worker_b
    _llm = ChatOpenAI(
        model=wb_cfg.name,
        temperature=wb_cfg.temperature,
        openai_api_key=api_key,
        openai_api_base=cfg.openrouter.base_url,
        default_headers={
            "HTTP-Referer": "https://github.com/daisysooyeon/SAP-ERP-AI-Agent",
            "X-Title": "SAP-ERP-AI-Agent Worker-B",
        },
    )
    return _llm


# ---------------------------------------------------------------------------
# ① 쿼리 추출 + Query Expansion 프롬프트
# ---------------------------------------------------------------------------

_QUERY_EXTRACTION_SYSTEM = """\
You are an expert at extracting and expanding search queries from B2B SAP ERP customer emails.

The email may contain an ERP transaction request (e.g., changing an order) AND a knowledge question about SAP.
Extract ONLY the knowledge/inquiry part, then generate multiple search query variants.

Rules:
1. Analyze the email and identify the core SAP knowledge question (ignore any ERP action requests).
2. If a knowledge question exists, set 'has_knowledge_question' to true and generate 3 search query variants:
   - 'query_primary': the most direct restatement of the question
   - 'query_variant_1': same intent, different phrasing or keywords
   - 'query_variant_2': a broader or more technical reformulation
3. If the email contains NO knowledge question, set 'has_knowledge_question' to false and leave all queries empty ("").

Output MUST be a valid JSON object:
{{
  "has_knowledge_question": true or false,
  "query_primary": "...",
  "query_variant_1": "...",
  "query_variant_2": "..."
}}
"""

_QUERY_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _QUERY_EXTRACTION_SYSTEM),
    ("human", "Email:\n\n{user_input}"),
])


def _extract_rag_queries(user_input: str) -> list[str]:
    """이메일에서 RAG 검색 쿼리를 최대 3개 추출 (Query Expansion). 질문이 없으면 빈 리스트 반환."""
    from langchain_core.output_parsers import JsonOutputParser
    try:
        llm = _get_llm()
        chain = _QUERY_EXTRACTION_PROMPT | llm | JsonOutputParser()
        result = chain.invoke({"user_input": user_input})

        if not result.get("has_knowledge_question"):
            return []

        queries = []
        for key in ("query_primary", "query_variant_1", "query_variant_2"):
            q = (result.get(key) or "").strip()
            if q:
                queries.append(q)

        logger.info("[worker_b] ① Extracted %d queries: %s", len(queries), queries)
        return queries
    except Exception as e:
        logger.error("[worker_b] Query extraction failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# ④ 답변 생성 프롬프트
# ---------------------------------------------------------------------------

_ANSWER_SYSTEM = """\
You are an SAP ERP expert assistant. Answer the user's question based ONLY on the provided context from SAP documentation.

Guidelines:
- Be concise and accurate.
- If the context does not contain enough information to answer, say: "I could not find sufficient information in the SAP documentation to answer this question."
- Do NOT make up information not present in the context.
- Cite the source document when possible (e.g., "According to TS460...").
"""

_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _ANSWER_SYSTEM),
    ("human", "Question: {query}\n\nContext:\n{context}"),
])


def _generate_answer(query: str, docs: list) -> str:
    """top-k 컨텍스트를 기반으로 LLM 답변 생성."""
    if not docs:
        return "No relevant SAP documentation found for this query."

    # 컨텍스트 조합 (source 정보 포함)
    context_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        context_parts.append(f"[{i}] (Source: {source})\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    try:
        llm = _get_llm()
        chain = _ANSWER_PROMPT | llm
        result = chain.invoke({"query": query, "context": context})
        answer = result.content.strip()
        logger.info("[worker_b] ④ Answer generated (%d chars)", len(answer))
        return answer
    except Exception as e:
        logger.error("[worker_b] Answer generation failed: %s", e)
        return f"Answer generation failed: {e}"


def _serialize_docs(docs: list) -> list[dict]:
    """LangChain Document 리스트를 JSON 직렬화 가능한 dict 리스트로 변환"""
    return [
        {
            "content":  doc.page_content,
            "source":   doc.metadata.get("source", ""),
            "chunk_id": doc.metadata.get("chunk_id", -1),
            "page":     doc.metadata.get("page", -1),
        }
        for doc in docs
    ]


# ---------------------------------------------------------------------------
# LangGraph 노드
# ---------------------------------------------------------------------------

def worker_b_node(state: AgentState) -> dict:
    """
    LangGraph Worker B 노드.

    Reads:
        state["user_input"]  — 원본 이메일
        state["rag_query"]   — 있으면 쿼리 추출 스킵

    Returns partial AgentState update:
        {
            "rag_query":      str,
            "retrieved_docs": list[dict],  # [{content, source, chunk_id}]
            "rag_answer":     str,
            "error_messages": list[str],
        }
    """
    logger.info("[worker_b] ══════════════ Worker B START ══════════════")

    user_input: str = state.get("user_input", "")
    email_context: dict | None = state.get("email_context")
    errors: list[str] = list(state.get("error_messages", []))

    # 전처리 결과가 "지식 질문 없음"이라고 명확히 신호하면, 비싼 LLM 쿼리 추출 단계를
    # 건너뛴다. preprocess_ok=False이거나 context 자체가 없으면 안전하게 기존 경로로 진행.
    if (email_context
            and email_context.get("preprocess_ok")
            and email_context.get("mentions_question") is False):
        logger.info("[worker_b] Preprocessor: mentions_question=False → RAG 단계 스킵")
        return {
            "rag_query":      "",
            "retrieved_docs": [],
            "rag_answer":     "",
            "error_messages": errors,
        }

    # 전처리가 만든 cleaned_body가 있으면 그것을 쿼리 추출 입력으로 사용 (인사·서명 노이즈
    # 제거 → 더 정확한 RAG query 추출). 없으면 원본 user_input 그대로.
    query_extraction_input = user_input
    if email_context and email_context.get("preprocess_ok") and email_context.get("cleaned_body"):
        query_extraction_input = email_context["cleaned_body"]

    # ① 쿼리 추출 (Query Expansion: 최대 3개)
    cached_query: str = state.get("rag_query", "")
    if cached_query:
        rag_queries = [cached_query]
    else:
        rag_queries = _extract_rag_queries(query_extraction_input)

    if not rag_queries:
        logger.warning("[worker_b] No RAG query extracted — skipping retrieval.")
        return {
            "rag_query": "",
            "retrieved_docs": [],
            "rag_answer": "",
            "error_messages": errors,
        }

    rag_query = rag_queries[0]  # 대표 쿼리 (답변 생성 + reranking 기준)

    # ② Hybrid Retrieval — 쿼리별 검색 후 중복 제거 병합
    logger.info("[worker_b] ② Hybrid retrieval (%d queries)", len(rag_queries))
    try:
        retriever = build_hybrid_retriever()
        seen_contents: set[str] = set()
        candidate_docs = []
        for q in rag_queries:
            results = retriever.invoke(q)
            for doc in results:
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    candidate_docs.append(doc)
        logger.info("[worker_b] ② Retrieved %d unique candidate docs", len(candidate_docs))
    except Exception as e:
        msg = f"worker_b: retrieval failed — {e}"
        logger.error("[worker_b] %s", msg)
        errors.append(msg)
        return {
            "rag_query": rag_query,
            "retrieved_docs": [],
            "rag_answer": "",
            "error_messages": errors,
        }

    # ③ Reranking (대표 쿼리 기준)
    logger.info("[worker_b] ③ Reranking %d docs …", len(candidate_docs))
    try:
        top_docs = rerank(rag_query, candidate_docs, queries=rag_queries)
        logger.info("[worker_b] ③ Reranked → top %d docs", len(top_docs))
    except Exception as e:
        logger.warning("[worker_b] Reranking failed: %s. Using top-k directly.", e)
        cfg = get_config()
        top_docs = candidate_docs[: cfg.rag.top_k_rerank]

    # state에 저장할 문서 메타 직렬화
    retrieved_docs_serialized = _serialize_docs(top_docs)

    # ④ 답변 생성
    logger.info("[worker_b] ④ Generating answer …")
    rag_answer = _generate_answer(rag_query, top_docs)

    logger.info("[worker_b] ══════════════ Worker B END ══════════════")
    return {
        "rag_query":      rag_query,
        "rag_queries":    rag_queries,
        "retrieved_docs": retrieved_docs_serialized,
        "rag_answer":     rag_answer,
        "error_messages": errors,
    }
