"""
src/main.py
LangGraph 그래프 진입점 — StateGraph 빌드 및 실행
"""
import uuid
import argparse

from src.logging_config import setup_logging
setup_logging()  # Must be called before any other src.* imports

from langgraph.graph import StateGraph, END
from langgraph.types import Send
from langgraph.checkpoint.sqlite import SqliteSaver

from src.config import get_config
from src.graph.state import AgentState
from src.graph.router import router_node
from src.graph.worker_a import worker_a_node
from src.graph.worker_b import worker_b_node
from src.graph.human_loop import human_loop_node
from src.graph.synthesizer import synthesizer_node


def build_graph():
    """
    LangGraph StateGraph 빌드
    
    분기 규칙:
      router → ACTION_ONLY → worker_a
      router → QA_ONLY    → worker_b
      router → BOTH       → worker_a + worker_b (Send API 병렬)
      worker_a → PENDING_APPROVAL → human_loop (interrupt)
      worker_a → 그 외           → synthesizer
      worker_b / human_loop       → synthesizer
      synthesizer                 → END
    """
    builder = StateGraph(AgentState)

    # 노드 등록
    builder.add_node("router", router_node)
    builder.add_node("worker_a", worker_a_node)
    builder.add_node("worker_b", worker_b_node)
    builder.add_node("human_loop", human_loop_node)
    builder.add_node("synthesizer", synthesizer_node)

    # 진입점
    builder.set_entry_point("router")

    # 라우터 → 분기 (BOTH일 때 Send API로 worker_a + worker_b 병렬 실행)
    def route_after_router(state: AgentState):
        intent = state["intent"]
        if intent == "ACTION_ONLY":
            return [Send("worker_a", state)]
        elif intent == "QA_ONLY":
            return [Send("worker_b", state)]
        else:  # BOTH
            return [Send("worker_a", state), Send("worker_b", state)]

    builder.add_conditional_edges("router", route_after_router, ["worker_a", "worker_b"])

    builder.add_conditional_edges(
        "worker_a",
        lambda state: (
            "human_loop"
            if state.get("erp_action_status") == "PENDING_APPROVAL"
            else "synthesizer"
        ),
    )

    builder.add_edge("worker_b", "synthesizer")
    builder.add_edge("human_loop", "synthesizer")
    builder.add_edge("synthesizer", END)

    # 영속 체크포인트 (interrupt 복구용)
    checkpoint_db = get_config().paths.checkpoint_db
    memory = SqliteSaver.from_conn_string(checkpoint_db)
    graph = builder.compile(checkpointer=memory, interrupt_before=["human_loop"])
    return graph


graph = build_graph()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAP ERP AI Agent")
    parser.add_argument("--input", required=True, help="이메일 텍스트 입력")
    args = parser.parse_args()

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {
            "user_input": args.input,
            "error_messages": [],
            "requires_human_approval": False,
            "human_approved": None,
        },
        config=config,
    )
    print("=== 최종 응답 ===")
    print(result.get("final_response", "응답 없음"))
