"""
src/api/server.py
FastAPI 서버: 에이전트 실행, 상태 조회, Slack 승인 처리, RAG 재인제스트
"""

import uuid
from fastapi import FastAPI
from src.api.schemas import RunRequest, RunResponse
from src.main import graph

app = FastAPI(title="SAP ERP AI Agent API", version="0.1.0")


@app.post("/api/run", response_model=RunResponse)
async def run_agent(request: RunRequest) -> RunResponse:
    """
    에이전트 실행 엔드포인트
    - interrupt 시 자동 일시 정지 (requires_approval=True)
    - thread_id로 상태 복구 가능
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "user_input": request.email_text,
        "error_messages": [],
        "requires_human_approval": False,
        "human_approved": None,
    }

    result = await graph.ainvoke(initial_state, config=config)

    return RunResponse(
        thread_id=thread_id,
        intent=result.get("intent"),
        erp_status=result.get("erp_action_status"),
        final_response=result.get("final_response"),
        requires_approval=result.get("requires_human_approval", False),
    )


@app.get("/api/status/{thread_id}")
async def get_status(thread_id: str) -> dict:
    """실행 상태 조회 — 체크포인트에서 현재 state 반환"""
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    if not state:
        return {"error": "thread_id를 찾을 수 없습니다."}
    return {"thread_id": thread_id, "state": state.values}


@app.get("/api/approve")
async def approve_action(thread_id: str, approved: bool) -> dict:
    """
    Slack 버튼 클릭 시 호출됨
    human_approved 업데이트 후 그래프 재개 (human_loop 노드 진입)
    """
    config = {"configurable": {"thread_id": thread_id}}
    graph.update_state(config, {"human_approved": approved})
    async for _ in graph.astream(None, config=config):
        pass
    return {"status": "processed", "approved": approved}


@app.post("/api/ingest")
async def ingest_docs() -> dict:
    """RAG 문서 추가/업데이트 트리거"""
    from src.rag.ingest import ingest_documents
    ingest_documents()
    return {"status": "ingest_complete"}
