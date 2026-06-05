"""
src/graph_builder.py
LangGraph StateGraph 팩토리 — 그래프 구조 정의 및 컴파일
"""
import uuid
import argparse

from src.logging_config import setup_logging
setup_logging()  # Must be called before any other src.* imports

from langgraph.graph import StateGraph, END
from langgraph.types import Send
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

from src.config import get_config
from src.graph.state import AgentState
from src.graph.router import router_node
from src.graph.worker_a import worker_a_node
from src.graph.worker_b import worker_b_node
from src.graph.human_loop import human_loop_node
from src.graph.synthesizer import synthesizer_node
from src.preprocess import preprocess_email


def _build_state_graph(*, hitl: bool = True) -> StateGraph:
    """StateGraph 구조만 빌드 (체크포인터 없음) — sync/async 공용.

    hitl=False 이면 human_loop 노드를 추가하지 않고 worker_a → synthesizer 직통.
    """
    builder = StateGraph(AgentState)

    builder.add_node("router", router_node)
    builder.add_node("worker_a", worker_a_node)
    builder.add_node("worker_b", worker_b_node)
    builder.add_node("synthesizer", synthesizer_node)

    if hitl:
        builder.add_node("human_loop", human_loop_node)

    builder.set_entry_point("router")

    def route_after_router(state: AgentState):
        intent = state["intent"]
        if intent == "ACTION_ONLY":
            return [Send("worker_a", state)]
        elif intent == "QA_ONLY":
            return [Send("worker_b", state)]
        else:  # BOTH
            return [Send("worker_a", state), Send("worker_b", state)]

    builder.add_conditional_edges("router", route_after_router, ["worker_a", "worker_b"])

    if hitl:
        _hitl_config_enabled = get_config().feature_flags.human_in_the_loop

        def route_after_worker_a(state: AgentState) -> str:
            if state.get("erp_action_status") == "PENDING_APPROVAL" and _hitl_config_enabled:
                return "human_loop"
            return "synthesizer"

        builder.add_conditional_edges("worker_a", route_after_worker_a)
        builder.add_edge("human_loop", "synthesizer")
    else:
        builder.add_edge("worker_a", "synthesizer")

    builder.add_edge("worker_b", "synthesizer")
    builder.add_edge("synthesizer", END)

    return builder


def build_graph():
    """SqliteSaver(sync)로 그래프 빌드 — CLI / 테스트 전용"""
    checkpoint_db = get_config().paths.checkpoint_db
    conn = sqlite3.connect(checkpoint_db, check_same_thread=False)
    memory = SqliteSaver(conn)
    return _build_state_graph().compile(checkpointer=memory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAP ERP AI Agent")
    parser.add_argument("--input", required=True, help="이메일 텍스트 입력")
    args = parser.parse_args()

    graph = build_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # LangGraph 진입 전 이메일 전처리 — 발신자/수신자/요청 사항/엔티티 구조화
    email_ctx = preprocess_email(args.input)

    result = graph.invoke(
        {
            "user_input":     args.input,
            "email_context":  email_ctx.model_dump(),
            "error_messages": [],
            "requires_human_approval": False,
            "human_approved": None,
        },
        config=config,
    )
    print("=== 최종 응답 ===")
    print(result.get("final_response", "응답 없음"))
