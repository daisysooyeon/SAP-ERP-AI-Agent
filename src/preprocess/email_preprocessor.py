"""
src/preprocess/email_preprocessor.py
LangGraph 진입 전 이메일 전처리 — 자연어 이메일 → 구조화된 EmailContext.

목적:
  현재 router / worker_a / worker_b / synthesizer 4개 노드가 각자 LLM 호출로
  같은 이메일을 다시 파싱한다(누가 보냈는지, 무엇을 요청하는지, 어떤 숫자가 있는지…).
  이 전처리를 한 번 수행해서 EmailContext에 담아두면:
    1) 다운스트림 노드의 LLM 호출이 짧은 hint를 받아 정확도/일관성 향상
    2) 같은 추론을 4번 반복하는 낭비 감소
    3) 디버깅 가시성 향상 (이메일 한 통이 어떻게 파싱됐는지 한곳에서 확인)

설계:
  - LangGraph 외부에서 동작 (graph_builder / api/server / eval 진입점에서 호출).
  - 결과는 AgentState["email_context"]에 dict로 적재. 노드들은 있으면 활용, 없으면
    기존 동작(원본 user_input 파싱)으로 자동 fallback — 하위호환.
  - 실패해도 그래프 흐름은 중단하지 말 것. EmailContext.preprocess_ok=False로 표시만.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.config import get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 구조화된 결과 스키마
# ---------------------------------------------------------------------------

class EmailContext(BaseModel):
    """전처리 결과 — LangGraph 노드들이 user_input과 함께 참고하는 구조화 정보."""

    # 원문 (그대로 보존 — 다운스트림 노드가 필요 시 참고)
    raw_text: str = Field(..., description="원본 이메일 텍스트 (수정 없음)")

    # 메타데이터 (LLM이 추출, 없으면 빈 문자열)
    sender_name:    str = Field("", description="발신자 이름 (서명/From 라인 등에서 추출). 모르면 빈 문자열.")
    sender_email:   str = Field("", description="발신자 이메일 주소. 모르면 빈 문자열.")
    sender_company: str = Field("", description="발신자 소속 회사명. 모르면 빈 문자열.")
    recipient:      str = Field("", description="수신자 (담당자/팀명). 모르면 빈 문자열.")
    subject:        str = Field("", description="제목. 본문에서 추정해도 됨.")
    language:       str = Field("en", description="이메일 언어 (en/ko/ja/zh/...). 기본 en.")

    # 정제된 본문 — 인사말/서명/Disclaimer 제외, 본문 핵심만
    cleaned_body: str = Field(..., description="인사·서명·면책조항을 제외한 본문 핵심")

    # 요청 요약 — 한두 문장
    request_summary: str = Field(
        ...,
        description="이메일이 무엇을 요청하는지 한두 문장 요약 (영어로). "
                    "예: 'Change quantity of item 10 on order 4500023456 to 100 units.'",
    )

    # 지식 질문만 분리·정규화 — RAG(worker_b) 검색 쿼리 입력으로 사용
    question_summary: str = Field(
        "",
        description="이메일에 SAP 지식/방법(how-to/정책/개념) 질문이 있으면, ERP 액션 맥락을 "
                    "모두 제거하고 하나의 명확하고 자기완결적인 영어 질문으로 재작성. 특정 주문/아이템 "
                    "번호나 수량 같은 트랜잭션 디테일은 빼고 일반화한다. 지식 질문이 없으면 빈 문자열. "
                    "예: 'How do I view the incompletion log for a sales order before creating an outbound delivery?'",
    )

    # ── 빠른 라우팅 힌트 ─────────────────────────────────────────────────────
    mentions_action: bool = Field(
        False,
        description="ERP 수정 요청(수량/날짜/취소/주소 변경 등)이 명시되어 있는가",
    )
    mentions_question: bool = Field(
        False,
        description="SAP/정책/매뉴얼에 대한 지식/문의 질문이 명시되어 있는가",
    )

    # ── 사전 추출된 핵심 엔티티 (다운스트림이 hint로 활용) ────────────────────
    order_ids: list[str] = Field(
        default_factory=list,
        description="이메일에 명시된 모든 영업 오더 번호(VBELN) 후보 — 원본 숫자 그대로, 패딩 없이",
    )
    item_nos: list[str] = Field(
        default_factory=list,
        description="이메일에 명시된 모든 아이템 번호(POSNR) 후보 — 원본 숫자 그대로",
    )

    # ── 메타 ───────────────────────────────────────────────────────────────
    preprocess_ok: bool = Field(True, description="전처리 성공 여부 (LLM 실패 시 False)")
    error: str = Field("", description="실패 사유 (preprocess_ok=False일 때만)")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a B2B email preprocessor for an SAP ERP support pipeline. Your job is to
read a single customer email and produce a structured summary that downstream
agents (router, ERP action extractor, RAG question extractor, reply composer)
will consume.

Extract the following fields from the email and output ONLY valid JSON matching
the required schema.

Field guidance:
  • sender_name / sender_email / sender_company:
      Look in the signature block ("Best regards, John Smith / john@acme.com /
      Acme Corp"), or in "From:" headers if present. If a field is not
      recoverable from the email, return an empty string. Do NOT invent.
  • recipient:
      The addressed party ("Hi Support Team", "Dear SAP Desk"). Empty if unclear.
  • subject:
      The Subject: line if present, otherwise a short topic phrase you infer
      from the body (under 80 chars).
  • language:
      Two-letter code: "en", "ko", "ja", "zh", "de", "es", "fr". Default "en".
  • cleaned_body:
      The substantive body — strip greetings ("Dear ..."), sign-offs ("Best
      regards, ..."), legal disclaimers, and "Sent from my iPhone"-style
      footers. Keep the actual request and any context the customer provides.
      Preserve order numbers, item numbers, quantities, dates exactly as
      written.
  • request_summary:
      ONE or TWO sentences in ENGLISH that describe what the customer wants.
      Be specific — include order number, item number, action, and quantities
      or dates if mentioned. Examples:
        "Change quantity of item 10 on order 4500023456 to 100 units."
        "Reduce quantity of item 20 on order 6105 by 50, and ask about the
         late-delivery penalty policy."
  • question_summary:
      If (and only if) the email contains an SAP knowledge / how-to / policy /
      concept question, restate THAT question as ONE clear, self-contained
      question in ENGLISH — optimized as a documentation search query. STRIP all
      transaction specifics (order/item numbers, quantities, dates) and any ERP
      action context; generalize to the underlying SAP concept. If there is no
      knowledge question, return an empty string. Examples:
        Email: "Reduce qty of item 20 on order 6105 by 50. Also, how do I run
                the backorder processing report to find unconfirmed orders?"
        question_summary: "How do I run the backorder processing report in SAP
                S/4HANA to identify unconfirmed sales orders?"
        Email: "Please update payment terms on order 5206 item 10 to Net 60."
        question_summary: ""   (pure action, no knowledge question)
  • mentions_action:
      true if the email asks to MODIFY anything in the ERP — change quantity,
      change delivery date, cancel an item, change shipping address, unblock a
      delivery, change carrier/batch/payment terms. false if it is a pure
      knowledge question.
  • mentions_question:
      true if the email asks about HOW SAP works / what a policy says / how to
      configure something / definitions — i.e. knowledge-base questions answered
      from documentation. false if it is purely a transaction request.
      mentions_action and mentions_question are NOT mutually exclusive — an
      email can have both.
  • order_ids:
      EVERY distinct order number (VBELN) mentioned. Copy digits EXACTLY as
      written — do NOT zero-pad, do NOT add or drop digits. Numbers may
      appear after "order", "order #", "sales order", "PO". Return as strings,
      no commas/dashes. Empty list if none.
  • item_nos:
      EVERY distinct item / line / position number (POSNR) mentioned. Copy
      digits EXACTLY. Numbers may appear after "item", "line", "position".
      Empty list if none.

CRITICAL — number handling:
  A single email often contains the order number, the item number, AND a
  quantity. They are DIFFERENT things. Map each to the correct field. Never
  put a quantity ("100 units", "50 pcs") into order_ids or item_nos.

Output ONLY valid JSON matching the schema. Set raw_text to the EXACT input
email (unchanged). Set preprocess_ok=true and error="" — these are only set to
false by the calling code on failure.
"""

_HUMAN_TEMPLATE = "Email to preprocess:\n\n{email}"

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    ("human",  _HUMAN_TEMPLATE),
])


# ---------------------------------------------------------------------------
# LLM factory (OpenRouter)
# ---------------------------------------------------------------------------

def _build_llm() -> ChatOpenAI:
    """전처리 LLM. configs.yaml의 models.preprocessor를 우선 사용하고,
    설정이 없으면 worker_a(이미 구조화 추출에 잘 동작)와 동일한 모델로 폴백."""
    cfg = get_config()

    # 동적 lookup — 새 설정 키가 없어도 동작하도록
    preproc_cfg = getattr(cfg.models, "preprocessor", None) or cfg.models.worker_a

    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set. Check your .env file.")

    return ChatOpenAI(
        model=preproc_cfg.name,
        temperature=preproc_cfg.temperature,
        openai_api_key=api_key,
        openai_api_base=cfg.openrouter.base_url,
        default_headers={
            "HTTP-Referer": "https://github.com/daisysooyeon/SAP-ERP-AI-Agent",
            "X-Title": "SAP ERP AI Agent - Preprocessor",
        },
    )


_llm: Optional[ChatOpenAI] = None
_chain = None


def _get_chain():
    """LLM 체인을 lazy하게 생성 — import 시점에 API 키 검증을 미루기 위함."""
    global _llm, _chain
    if _chain is None:
        _llm = _build_llm()
        _chain = _PROMPT | _llm.with_structured_output(EmailContext)
    return _chain


# ---------------------------------------------------------------------------
# Regex 기반 backstop entity 추출 (LLM 실패/누락 보조)
# ---------------------------------------------------------------------------

_ORDER_RE = re.compile(
    r"(?:order|order\s*#|sales\s+order|PO)\s*[:#]?\s*(\d{3,10})",
    re.IGNORECASE,
)
_ITEM_RE = re.compile(
    r"(?:item|line|position)\s*[:#]?\s*(\d{1,6})",
    re.IGNORECASE,
)


def _regex_entities(text: str) -> tuple[list[str], list[str]]:
    """LLM이 실패하거나 빠뜨릴 때를 대비한 정규식 backstop. 중복 제거."""
    order_ids = list(dict.fromkeys(_ORDER_RE.findall(text)))
    item_nos  = list(dict.fromkeys(_ITEM_RE.findall(text)))
    return order_ids, item_nos


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_email(raw_email: str) -> EmailContext:
    """이메일을 LLM 한 번 호출로 구조화된 EmailContext로 변환.

    LLM이 실패해도 EmailContext.preprocess_ok=False로 표시하고 정규식 backstop으로
    최소한의 정보를 채워 반환한다. 호출자(graph entry point)는 그냥 통과시키면
    되고, 다운스트림 노드는 빈 필드를 보면 기존 user_input 파싱으로 폴백한다.
    """
    raw_email = raw_email or ""
    logger.info("[preprocess] Preprocessing email (%d chars)…", len(raw_email))

    try:
        chain = _get_chain()
        result: EmailContext = chain.invoke({"email": raw_email})

        # LLM이 raw_text를 비우거나 변형했을 수 있으므로 결정론적으로 덮어쓴다
        result.raw_text = raw_email

        # LLM이 엔티티를 빠뜨렸으면 정규식으로 보강 (덮어쓰지 않고 union)
        rx_orders, rx_items = _regex_entities(raw_email)
        if not result.order_ids and rx_orders:
            result.order_ids = rx_orders
        if not result.item_nos and rx_items:
            result.item_nos = rx_items

        logger.info(
            "[preprocess] OK | sender=%r action=%s question=%s "
            "orders=%s items=%s lang=%s",
            result.sender_name, result.mentions_action,
            result.mentions_question, result.order_ids, result.item_nos,
            result.language,
        )
        return result

    except Exception as e:
        # 실패해도 흐름은 끊지 않는다 — backstop으로 최소 정보 채워 반환
        logger.error("[preprocess] FAILED: %s — falling back to regex backstop only", e)
        rx_orders, rx_items = _regex_entities(raw_email)
        return EmailContext(
            raw_text=raw_email,
            cleaned_body=raw_email,         # 원본을 그대로 유지
            request_summary="",
            order_ids=rx_orders,
            item_nos=rx_items,
            preprocess_ok=False,
            error=f"{type(e).__name__}: {e}",
        )
