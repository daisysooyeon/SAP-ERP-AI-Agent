"""
src/slack/notifier.py
Send ERP update approval requests via Slack Incoming Webhook (including DMs)

Setup:
  Set SLACK_WEBHOOK_URL in .env
  (Slack App → Incoming Webhooks → "Add New Webhook to Workspace" → select DM channel)
"""

import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 메시지 포맷 빌더
# ---------------------------------------------------------------------------

def _build_action_summary(action: dict) -> str:
    """Build a summary text from the action dict"""
    action_type = action.get("action_type", "OTHER")
    order_id    = action.get("order_id", "-")
    item_no     = action.get("item_no", "-")

    if action_type == "CHANGE_QTY":
        detail = f"Quantity change  →  *{action.get('new_quantity', '-')} units*"
    elif action_type == "CHANGE_DATE":
        detail = f"Delivery date change  →  *{action.get('new_date', '-')}*"
    elif action_type == "CANCEL_ITEM":
        detail = "Item *cancellation* requested"
    elif action_type == "CHANGE_ADDR":
        detail = f"Delivery address change  →  {action.get('new_address', '-')}"
    else:
        detail = f"Other change (`{action_type}`)"

    return (
        f"*🔔 ERP Update Approval Required*\n"
        f"• Order No. : `{order_id}`\n"
        f"• Item No.  : `{item_no}`\n"
        f"• Change    : {detail}"
    )


def _build_message(action: dict, thread_id: str, server_base_url: str) -> dict:
    """Build a Slack Block Kit message payload"""
    approve_url = f"{server_base_url}/api/approve?thread_id={thread_id}&approved=true"
    reject_url  = f"{server_base_url}/api/approve?thread_id={thread_id}&approved=false"

    return {
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": _build_action_summary(action),
                },
            },
            {"type": "divider"},
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"🔑 Thread ID: `{thread_id}`",
                    }
                ],
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "✅ Approve", "emoji": True},
                        "style": "primary",
                        "url": approve_url,
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "❌ Reject", "emoji": True},
                        "style": "danger",
                        "url": reject_url,
                    },
                ],
            },
        ]
    }


# ---------------------------------------------------------------------------
# 발송 함수 (동기 + 비동기)
# ---------------------------------------------------------------------------

def send_approval_request(
    action: dict,
    thread_id: str,
    server_base_url: str = "http://localhost:8000",
) -> bool:
    """
    Sync version: Send an ERP update approval request to Slack.

    Returns:
        True  — message sent successfully
        False — SLACK_WEBHOOK_URL not set, or send failed
    """
    webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")
    if not webhook_url:
        logger.warning("[slack] SLACK_WEBHOOK_URL is not set. Skipping message.")
        return False

    message = _build_message(action, thread_id, server_base_url)
    try:
        resp = httpx.post(webhook_url, json=message, timeout=5.0)
        resp.raise_for_status()
        logger.info("[slack] Approval request sent (thread_id=%s)", thread_id)
        return True
    except Exception as e:
        logger.error("[slack] Failed to send message: %s", e)
        return False


async def send_approval_request_async(
    action: dict,
    thread_id: str,
    server_base_url: str = "http://localhost:8000",
) -> bool:
    """
    Async version: for use inside FastAPI request handlers.

    Returns:
        True  — message sent successfully
        False — SLACK_WEBHOOK_URL not set, or send failed
    """
    webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")
    if not webhook_url:
        logger.warning("[slack] SLACK_WEBHOOK_URL is not set. Skipping message.")
        return False

    message = _build_message(action, thread_id, server_base_url)
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(webhook_url, json=message)
            resp.raise_for_status()
            logger.info("[slack] Approval request sent (thread_id=%s)", thread_id)
            return True
    except Exception as e:
        logger.error("[slack] Failed to send message: %s", e)
        return False
