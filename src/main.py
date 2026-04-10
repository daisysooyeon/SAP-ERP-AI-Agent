"""
src/main.py
LangGraph 그래프 진입점 — StateGraph 빌드 및 실행
"""
from src.logging_config import setup_logging
setup_logging()  # Must be called before any other src.* imports

from langgraph.graph import StateGraph, END
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

    # 라우터 → 분기
    builder.add_conditional_edges(
        "router",
        lambda state: state["intent"],
        {
            "ACTION_ONLY": "worker_a",
            "QA_ONLY": "worker_b",
            "BOTH": "worker_a",  # BOTH: Send API로 worker_b 병렬 실행 (TODO)
        },
    )

    # Worker A → Human-in-the-Loop 또는 합성
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
    import argparse

    parser = argparse.ArgumentParser(description="SAP ERP AI Agent")
    parser.add_argument("--input", required=True, help="이메일 텍스트 입력")
    args = parser.parse_args()

    import uuid

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
