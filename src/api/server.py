"""
src/api/server.py
FastAPI server: agent execution + Slack approval integration + Human-in-the-Loop resume

Endpoints:
  GET  /api/health                      — health check + public URL
  POST /api/run                         — run agent (sends Slack alert on interrupt)
  GET  /api/status/{thread_id}          — query current execution state
  GET  /api/approve?thread_id=&approved=— handle Slack button click → resume graph
  POST /api/ingest                      — re-ingest RAG documents

ngrok integration:
  Set NGROK_AUTHTOKEN in .env to automatically open an ngrok tunnel on startup.
  The public URL is used in Slack button links automatically.
  If not set, falls back to SERVER_BASE_URL env var (default: http://localhost:8000).
"""

import logging
import os
import uuid

from dotenv import load_dotenv
load_dotenv()
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from src.api.schemas import ApproveResponse, RunRequest, RunResponse
from src.slack.notifier import send_approval_request_async

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public URL management (ngrok or SERVER_BASE_URL)
# ---------------------------------------------------------------------------

_public_url: str = os.getenv("SERVER_BASE_URL", "http://localhost:8000")


def get_public_url() -> str:
    return _public_url


# ---------------------------------------------------------------------------
# App lifespan (ngrok auto-start / teardown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _public_url

    ngrok_token = os.getenv("NGROK_AUTHTOKEN", "")

    if ngrok_token:
        try:
            from pyngrok import conf, ngrok as pyngrok_client
            conf.get_default().auth_token = ngrok_token
            tunnel = pyngrok_client.connect(8000, proto="http")
            _public_url = tunnel.public_url
            logger.info("[server] ngrok tunnel active: %s", _public_url)
            print(f"\n  PUBLIC URL  : {_public_url}")
            print(f"  Swagger UI  : {_public_url}/docs\n")
        except ImportError:
            logger.warning("[server] pyngrok not installed -- run 'pip install pyngrok'.")
        except Exception as e:
            logger.error("[server] Failed to start ngrok: %s", e)
    else:
        logger.info("[server] SERVER_BASE_URL: %s (ngrok disabled)", _public_url)
        print(f"\n  PUBLIC URL  : {_public_url}  (NGROK_AUTHTOKEN not set)")
        print( "      Add NGROK_AUTHTOKEN to .env for external access.\n")

    # Build graph with AsyncSqliteSaver (required for async FastAPI endpoints)
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from src.main import _build_state_graph
    from src.config import get_config

    checkpoint_db = get_config().paths.checkpoint_db
    async with aiosqlite.connect(checkpoint_db) as aio_conn:
        memory = AsyncSqliteSaver(aio_conn)
        app.state.graph = _build_state_graph().compile(checkpointer=memory)

        yield

    # Teardown: close ngrok tunnel
    if ngrok_token:
        try:
            from pyngrok import ngrok as pyngrok_client
            pyngrok_client.kill()
            logger.info("[server] ngrok tunnel closed")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# FastAPI 앱
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SAP ERP AI Agent API",
    version="0.1.0",
    description="SAP ERP AI Agent — Human-in-the-Loop approval workflow",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health() -> dict:
    """Health check — server status and current public URL"""
    return {
        "status": "ok",
        "public_url": get_public_url(),
    }


@app.post("/api/run", response_model=RunResponse)
async def run_agent(request: RunRequest) -> RunResponse:
    """
    Agent execution endpoint

    Flow:
      1. Run LangGraph graph
      2. Auto-pauses on PENDING_APPROVAL (interrupt_before=["human_loop"])
      3. Sends Slack DM approval request when paused
      4. Returns thread_id (resume later via /api/approve)
    """
    graph = app.state.graph
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "user_input": request.email_text,
        "error_messages": [],
        "requires_human_approval": False,
        "human_approved": None,
    }

    result = await graph.ainvoke(initial_state, config=config)

    requires_approval = result.get("requires_human_approval", False)
    erp_action        = result.get("erp_action")

    # ── Send Slack approval request ───────────────────────────────────────────
    if requires_approval and erp_action:
        sent = await send_approval_request_async(
            action=erp_action,
            thread_id=thread_id,
            server_base_url=get_public_url(),
        )
        if sent:
            logger.info("[api/run] Slack approval request sent — thread_id=%s", thread_id)
        else:
            logger.warning("[api/run] Slack send failed (check SLACK_WEBHOOK_URL)")

    return RunResponse(
        thread_id=thread_id,
        intent=result.get("intent"),
        erp_status=result.get("erp_action_status"),
        final_response=result.get("final_response"),
        requires_approval=requires_approval,
    )


@app.get("/api/status/{thread_id}")
async def get_status(thread_id: str) -> dict:
    """Fetch current state from the checkpoint store"""
    graph = app.state.graph
    config = {"configurable": {"thread_id": thread_id}}
    state = await graph.aget_state(config)

    if not state or not state.values:
        raise HTTPException(status_code=404, detail="thread_id not found.")

    return {
        "thread_id":         thread_id,
        "intent":            state.values.get("intent"),
        "erp_action_status": state.values.get("erp_action_status"),
        "erp_action":        state.values.get("erp_action"),
        "error_messages":    state.values.get("error_messages", []),
    }


@app.get("/api/approve", response_model=ApproveResponse)
async def approve_action(thread_id: str, approved: bool) -> ApproveResponse:
    """
    Called when the Slack Approve / Reject button is clicked.

    Flow:
      1. Verify the thread checkpoint exists
      2. Update human_approved in the checkpoint
      3. Resume graph → runs human_loop_node
         - approved=True  → SQLite UPDATE → erp_action_status=SUCCESS
         - approved=False → no DB change  → erp_action_status=REJECTED
      4. Return final status (visible in browser after button click)
    """
    graph = app.state.graph
    config = {"configurable": {"thread_id": thread_id}}

    # 1. Verify checkpoint exists
    state = await graph.aget_state(config)
    if not state or not state.values:
        raise HTTPException(
            status_code=404,
            detail=f"thread_id '{thread_id}' not found.",
        )

    current_status = state.values.get("erp_action_status")
    if current_status != "PENDING_APPROVAL":
        # Already processed
        return ApproveResponse(
            thread_id=thread_id,
            approved=approved,
            final_status=current_status or "UNKNOWN",
            message=f"This request has already been processed. Current status: {current_status}",
        )

    # 2. Resume graph via Command(resume=...) — LangGraph 1.x interrupt() 방식
    from langgraph.types import Command
    async for _ in graph.astream(Command(resume=approved), config=config):
        pass

    # 4. Fetch final state
    final_state   = await graph.aget_state(config)
    final_status  = "UNKNOWN"
    errors: list  = []

    if final_state and final_state.values:
        final_status = final_state.values.get("erp_action_status", "UNKNOWN")
        errors       = final_state.values.get("error_messages", [])

    action_label = "approved" if approved else "rejected"
    logger.info(
        "[api/approve] %s — thread_id=%s  final_status=%s",
        action_label, thread_id, final_status,
    )

    return ApproveResponse(
        thread_id=thread_id,
        approved=approved,
        final_status=final_status,
        message=f"ERP update {action_label}. Final status: {final_status}",
        errors=errors,
    )


@app.post("/api/ingest")
async def ingest_docs() -> dict:
    """Trigger re-ingest of RAG documents"""
    from src.rag.ingest import ingest_documents
    ingest_documents()
    return {"status": "ingest_complete"}
