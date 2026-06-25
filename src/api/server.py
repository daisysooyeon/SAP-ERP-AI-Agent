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

import asyncio
import json
import logging
import os
import uuid
from urllib.parse import parse_qs

from dotenv import load_dotenv
load_dotenv()
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from src.api.schemas import ApproveResponse, RunRequest, RunResponse
from src.preprocess import preprocess_email
from src.slack.notifier import (
    open_reason_modal,
    send_approval_request_async,
    update_message_via_response_url,
    verify_slack_signature,
)

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
            # 고정 도메인이 지정되면 항상 같은 공개 URL로 터널을 연다.
            # (Slack Interactivity Request URL을 매번 다시 입력할 필요가 없어짐)
            ngrok_domain = os.getenv("NGROK_DOMAIN", "").strip()
            if ngrok_domain:
                tunnel = pyngrok_client.connect(8000, proto="http", domain=ngrok_domain)
            else:
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
    from src.graph_builder import _build_state_graph
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

# CORS: allow the Vercel static demo page (different origin) to call this API.
# 시연용으로 전체 허용. 운영 시 Vercel 도메인으로 제한 가능.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """데모 프론트엔드(web/index.html)를 동일 출처로 서빙 → 단일 배포 링크로 UI+API 제공."""
    from pathlib import Path
    return FileResponse(Path(__file__).resolve().parents[2] / "web" / "index.html")


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

    # LangGraph 진입 전 이메일 전처리 (블로킹 동기 호출 — 다른 비동기 작업과 동시에 돌릴
    # 필요는 없으나 이벤트 루프 점유 우려가 있으면 to_thread로 옮길 것). 실패해도 흐름은
    # 끊기지 않고 preprocess_ok=False 로 표시될 뿐이다.
    email_ctx = await asyncio.to_thread(preprocess_email, request.email_text)

    initial_state = {
        "user_input":             request.email_text,
        "email_context":          email_ctx.model_dump(),
        "error_messages":         [],
        "requires_human_approval": False,
        "human_approved":         None,
    }

    result = await graph.ainvoke(initial_state, config=config)

    requires_approval = result.get("requires_human_approval", False)
    erp_action        = result.get("erp_action")

    # ── Send Slack approval request ───────────────────────────────────────────
    if requires_approval and erp_action:
        sent = await send_approval_request_async(
            action=erp_action,
            thread_id=thread_id,
        )
        if sent:
            logger.info("[api/run] Slack approval request sent - thread_id=%s", thread_id)
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
        "final_response":    state.values.get("final_response"),
        "requires_approval": state.values.get("requires_human_approval", False),
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
    #    human_loop_node expects {"approved": bool, "reason": str|None}.
    from langgraph.types import Command
    async for _ in graph.astream(
        Command(resume={"approved": approved, "reason": None}), config=config
    ):
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


# ---------------------------------------------------------------------------
# Slack Interactivity (interactive buttons + rejection-reason modal)
# ---------------------------------------------------------------------------

async def _resume_graph(thread_id: str, approved: bool, reason: str | None) -> str:
    """
    Resume a paused graph with the human decision and return the final status.

    Guards against double-processing: if the thread is no longer PENDING_APPROVAL,
    the current status is returned without resuming again.
    """
    graph  = app.state.graph
    config = {"configurable": {"thread_id": thread_id}}

    state = await graph.aget_state(config)
    if not state or not state.values:
        logger.warning("[slack/actions] thread_id '%s' not found.", thread_id)
        return "NOT_FOUND"

    current = state.values.get("erp_action_status")
    if current != "PENDING_APPROVAL":
        logger.info("[slack/actions] thread_id '%s' already processed (%s).", thread_id, current)
        return current or "UNKNOWN"

    from langgraph.types import Command
    async for _ in graph.astream(
        Command(resume={"approved": approved, "reason": reason}), config=config
    ):
        pass

    final = await graph.aget_state(config)
    return (final.values.get("erp_action_status") if final and final.values else "UNKNOWN") or "UNKNOWN"


async def _resume_and_notify(
    thread_id: str, approved: bool, reason: str | None, response_url: str
) -> None:
    """
    Background task: resume the graph (DB update + email synthesis can take a few
    seconds — longer than Slack's 3s ack window) and then update the original Slack
    message via response_url. Run via asyncio.create_task so the HTTP handler can
    ack Slack immediately.
    """
    try:
        status = await _resume_graph(thread_id, approved, reason)
        if approved:
            text = f"✅ *Approved* — final status: *{status}*\n🔑 `{thread_id}`"
        else:
            reason_line = f"\n> _{reason}_" if reason else "  _(no reason provided)_"
            text = f"❌ *Rejected* — final status: *{status}*.{reason_line}\n🔑 `{thread_id}`"
        await update_message_via_response_url(response_url, text)
    except Exception as e:
        logger.error("[slack/actions] resume/notify failed for %s: %s", thread_id, e)
        await update_message_via_response_url(
            response_url, f"⚠️ Processing error for `{thread_id}`: {e}"
        )


@app.post("/slack/actions")
async def slack_actions(request: Request):
    """
    Slack Interactivity endpoint (Request URL = https://<host>/slack/actions).

    Handles two payload types:
      • block_actions  — Approve/Reject button clicks
          - erp_approve → resume graph (approved) in background, ack immediately
          - erp_reject  → open the rejection-reason modal (views.open)
      • view_submission — modal "Submit Rejection"
          - resume graph (rejected, with optional reason) in background, close modal
    """
    raw_body  = await request.body()
    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    signature = request.headers.get("X-Slack-Signature", "")

    if not verify_slack_signature(raw_body, timestamp, signature):
        raise HTTPException(status_code=401, detail="Invalid Slack signature.")

    # Slack sends application/x-www-form-urlencoded with a single `payload` field.
    # Parse the raw body directly so we don't depend on python-multipart.
    parsed = parse_qs(raw_body.decode("utf-8"))
    payload_raw = (parsed.get("payload") or [None])[0]
    if not payload_raw:
        raise HTTPException(status_code=400, detail="Missing Slack payload.")
    try:
        payload = json.loads(payload_raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Malformed Slack payload.")

    ptype = payload.get("type")

    # ── Button click ────────────────────────────────────────────────────────
    if ptype == "block_actions":
        action       = (payload.get("actions") or [{}])[0]
        action_id    = action.get("action_id", "")
        thread_id    = action.get("value", "")
        response_url = payload.get("response_url", "")

        if action_id == "erp_approve":
            asyncio.create_task(
                _resume_and_notify(thread_id, approved=True, reason=None, response_url=response_url)
            )
            await update_message_via_response_url(
                response_url, f"⏳ Approving… processing ERP update for `{thread_id}`."
            )
            return Response(status_code=200)

        if action_id == "erp_reject":
            trigger_id = payload.get("trigger_id", "")
            meta = json.dumps({"thread_id": thread_id, "response_url": response_url})
            await open_reason_modal(trigger_id, meta)
            return Response(status_code=200)

        return Response(status_code=200)

    # ── Modal submit (rejection reason) ───────────────────────────────────────
    if ptype == "view_submission":
        view = payload.get("view", {})
        try:
            meta = json.loads(view.get("private_metadata") or "{}")
        except json.JSONDecodeError:
            meta = {}
        thread_id    = meta.get("thread_id", "")
        response_url = meta.get("response_url", "")

        reason = ""
        try:
            reason = (
                view["state"]["values"]["reason_block"]["reason_input"].get("value") or ""
            )
        except (KeyError, TypeError):
            reason = ""
        reason = reason.strip()

        asyncio.create_task(
            _resume_and_notify(
                thread_id, approved=False, reason=(reason or None), response_url=response_url
            )
        )
        # Empty body → Slack closes the modal.
        return JSONResponse(content={})

    return Response(status_code=200)


@app.post("/api/ingest")
async def ingest_docs() -> dict:
    """Trigger re-ingest of RAG documents"""
    from src.rag.ingest import ingest_documents
    ingest_documents()
    return {"status": "ingest_complete"}
