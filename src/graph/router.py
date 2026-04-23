"""
src/graph/router.py
Orchestrator: Email intent classification node (ACTION_ONLY / QA_ONLY / BOTH)

Flow:
  1. Build ChatPromptTemplate with system prompt (few-shot examples included)
  2. Bind Pydantic RouterOutput schema to LLM as structured output
  3. Invoke the chain with the user's email text
  4. Return {"intent": ..., "error_messages": [...]} to update AgentState
"""

import os
import logging
import time
from typing import Literal

from dotenv import load_dotenv
load_dotenv()  # .env 파일의 환경변수를 os.environ에 로드

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from src.config import get_config
from src.graph.state import AgentState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Structured-output schema
# ---------------------------------------------------------------------------

class RouterOutput(BaseModel):
    """JSON structured output schema for the router LLM."""

    intent: Literal["ACTION_ONLY", "QA_ONLY", "BOTH"] = Field(
        description=(
            "Classified intent of the email. "
            "ACTION_ONLY = ERP modification only; "
            "QA_ONLY = policy/regulation question only; "
            "BOTH = ERP modification AND policy question present."
        )
    )
    reasoning: str = Field(
        description="Step-by-step chain-of-thought reasoning that justifies the chosen intent label."
    )


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

ROUTER_SYSTEM_PROMPT = """\
You are an expert at analyzing B2B sales emails in the context of SAP ERP operations.

Your task is to read the customer's email and classify it into **exactly one** of the following intent labels:

  - ACTION_ONLY  : The email contains ONLY requests to modify or query ERP data
                   (e.g., change order quantity, change delivery date, cancel an item).
  - QA_ONLY      : The email contains ONLY questions about internal policies or regulations
                   (e.g., penalty clauses, return policy, surcharge conditions).
                   For demonstration, there will be questions about the concepts of the SAP ERP system.
  - BOTH         : The email contains BOTH an ERP modification/query request AND a policy/regulation question.

Classification rules:
  1. If ANY ERP action request is present together with ANY policy question → BOTH.
  2. If ONLY ERP action requests are present (no policy questions) → ACTION_ONLY.
  3. If ONLY policy questions are present (no ERP action requests) → QA_ONLY.
  4. Greetings, sign-offs, and pleasantries are ignored for classification.
  5. Provide your reasoning step by step BEFORE stating the final label.

────────────────────────────────────────
FEW-SHOT EXAMPLES
────────────────────────────────────────

[Example 1]
Email: "Please change the quantity of PO-2024031200 to 500 units.
        Also, what are the additional charges for express delivery?"
Intent: BOTH
Reasoning: The email requests a quantity update (ERP action) AND asks about express delivery surcharges (policy question). Both categories are present → BOTH.

[Example 2]
Email: "Kindly update the delivery date for order 4500012345, item 000010 from March 25 to April 1."
Intent: ACTION_ONLY
Reasoning: The email solely requests a delivery date change on a specific sales order item. No policy question is present → ACTION_ONLY.

[Example 3]
Email: "Could you please explain the penalty clause that applies when we cancel an order after shipment?"
Intent: QA_ONLY
Reasoning: The email only asks about the penalty policy for post-shipment cancellation. No ERP modification is requested → QA_ONLY.

[Example 4]
Email: "We would like to cancel item 000020 on order 4500099871.
        Please also let us know the return policy for defective items."
Intent: BOTH
Reasoning: Cancelling an order item is an ERP action. Asking about the return policy is a policy question. Both are present → BOTH.

[Example 5]
Email: "What is the lead time policy for rush orders placed after the cutoff time?"
Intent: QA_ONLY
Reasoning: The email is entirely a policy inquiry about rush-order lead times. No ERP change is requested → QA_ONLY.

[Example 6]
Email: "Please update the requested quantity for order 4500067890, line 000030 to 1,200 units."
Intent: ACTION_ONLY
Reasoning: A single quantity change request with no policy or regulation question → ACTION_ONLY.

[Example 7]
Email: "I need to reduce the order quantity of PO-2024056789 from 800 to 600.
        In addition, can you tell me under what conditions we qualify for a volume discount?"
Intent: BOTH
Reasoning: Quantity reduction is an ERP action; the volume discount inquiry is a policy question. Both categories are present → BOTH.

[Example 8]
Email: "What is the difference between forward and backward scheduling?"
Intent: QA_ONLY
Reasoning: The email contains only a question about the scheduling concepts. No ERP action is requested → QA_ONLY.

[Example 9]
Email: "Please change the delivery address for sales order 4500034512 to our new warehouse in Incheon."
Intent: ACTION_ONLY
Reasoning: Updating a delivery address is an ERP modification request. No policy question is present → ACTION_ONLY.

[Example 10]
Email: "Hi, we want to move the delivery date of order 4500023456 to May 10th.
        Also, is there a late delivery penalty we should be aware of?"
Intent: BOTH
Reasoning: The delivery date change is an ERP action; the late delivery penalty question is a policy inquiry. Both are present → BOTH.

[Example 11]
Email: "Can you clarify how I can define copying rules?"
Intent: QA_ONLY
Reasoning: A pure concept related question about copying rules. No ERP transaction requested → QA_ONLY.

[Example 12]
Email: "Please cancel item 000010 on order 4500078901.
        What is the standard cancellation fee in this case?"
Intent: BOTH
Reasoning: The cancellation request is an ERP action; asking about the cancellation fee is a policy question. Both are present → BOTH.

────────────────────────────────────────
Now classify the following email. Output ONLY valid JSON matching the required schema.
────────────────────────────────────────
"""

ROUTER_HUMAN_TEMPLATE = "Email to classify:\n\n{user_input}"

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", ROUTER_SYSTEM_PROMPT),
        ("human", ROUTER_HUMAN_TEMPLATE),
    ]
)

# ---------------------------------------------------------------------------
# LLM factory  (swap out get_llm("router") once implemented)
# ---------------------------------------------------------------------------

def _build_llm():
    """
    Instantiate the router LLM based on configs.yaml `models.router.provider`.

    provider = "ollama"      → ChatOllama (local, no API key needed)
    provider = "openrouter"  → ChatOpenAI pointed at OpenRouter endpoint
    provider = "openai"      → ChatOpenAI pointed at OpenAI endpoint
    """
    cfg = get_config()
    router_cfg = cfg.models.router

    if router_cfg.provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set in .env")
        return ChatOpenAI(
            model=router_cfg.name,
            temperature=router_cfg.temperature,
            openai_api_key=api_key,
        )

    if router_cfg.provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY is not set. "
                "Add it to your .env file: OPENROUTER_API_KEY=sk-or-..."
            )
        return ChatOpenAI(
            model=router_cfg.name,
            temperature=router_cfg.temperature,
            openai_api_key=api_key,
            openai_api_base=cfg.openrouter.base_url,
            default_headers={
                "HTTP-Referer": "https://github.com/daisysooyeon/SAP-ERP-AI-Agent",
                "X-Title": "SAP ERP AI Agent",
            },
        )

    # default: Ollama
    return ChatOllama(
        base_url=cfg.ollama.base_url,
        model=router_cfg.name,
        temperature=router_cfg.temperature,
    )


# Build once at import time and reuse across invocations.
_llm = _build_llm()
_chain = _PROMPT | _llm.with_structured_output(RouterOutput)


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def router_node(state: AgentState) -> dict:
    """
    LangGraph router node — classifies email intent and updates AgentState.

    Reads:
        state["user_input"]  – raw email text

    Returns a partial AgentState update:
        {
            "intent":         "ACTION_ONLY" | "QA_ONLY" | "BOTH",
            "error_messages": [...],   # appended on failure
        }
    """
    user_input: str = state["user_input"]
    errors: list[str] = list(state.get("error_messages", []))

    logger.info("[router_node] Classifying email intent …")
    logger.debug("[router_node] Input: %s", user_input[:200])

    # Retry with exponential backoff on 429 Rate Limit errors
    max_retries = 5
    wait_secs = 5  # 5 → 10 → 20 → 40 → 80

    for attempt in range(1, max_retries + 1):
        try:
            result: RouterOutput = _chain.invoke({"user_input": user_input})
            break  # 성공 시 루프 탈출
        except Exception as exc:
            err_str = str(exc)
            # 429 Rate Limit인 경우 재시도
            if "429" in err_str and attempt < max_retries:
                logger.warning(
                    "[router_node] Rate limited (429). Attempt %d/%d — waiting %ds …",
                    attempt, max_retries, wait_secs,
                )
                time.sleep(wait_secs)
                wait_secs *= 2  # 지수 백오프
                continue
            # 429 외 에러이거나 재시도 초과 시 실패 처리
            logger.error("[router_node] LLM call failed: %s", exc, exc_info=True)
            errors.append(f"router_node error: {exc}")
            return {
                "intent": None,
                "error_messages": errors,
            }

    logger.info("[router_node] intent=%s | reasoning=%s", result.intent, result.reasoning[:120])

    return {
        "intent": result.intent,
        "error_messages": errors,
    }
