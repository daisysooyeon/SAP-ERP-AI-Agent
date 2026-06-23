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


def _build_message(action: dict, thread_id: str) -> dict:
    """
    Build a Slack Block Kit message payload with INTERACTIVE buttons.

    The buttons carry an ``action_id`` and ``value=thread_id`` (no ``url``), so
    clicking them triggers an interactivity callback to ``/slack/actions`` instead
    of opening a browser link. ``erp_approve`` resumes the graph immediately;
    ``erp_reject`` opens a modal asking for an (optional) rejection reason.

    Requires the Slack App to have Interactivity enabled with the Request URL
    pointed at ``https://<public-host>/slack/actions``.
    """
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
                "block_id": "erp_approval_actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "✅ Approve", "emoji": True},
                        "style": "primary",
                        "action_id": "erp_approve",
                        "value": thread_id,
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "❌ Reject", "emoji": True},
                        "style": "danger",
                        "action_id": "erp_reject",
                        "value": thread_id,
                    },
                ],
            },
        ]
    }


# ---------------------------------------------------------------------------
# Slack Interactivity helpers (signature verification, modal, message update)
# ---------------------------------------------------------------------------

import hashlib
import hmac
import time


def verify_slack_signature(body: bytes, timestamp: str, signature: str) -> bool:
    """
    Verify an incoming Slack request using the signing secret.

    Slack signs each request as:
        v0={HMAC_SHA256(signing_secret, f"v0:{timestamp}:{raw_body}")}

    Returns True only if the signature matches and the timestamp is recent
    (within 5 minutes — replay protection).
    """
    signing_secret = os.getenv("SLACK_SIGNING_SECRET", "")
    if not signing_secret or not timestamp or not signature:
        logger.warning("[slack] Signature verification skipped — missing secret/headers.")
        return False

    try:
        if abs(time.time() - int(timestamp)) > 60 * 5:
            logger.warning("[slack] Stale request timestamp — possible replay.")
            return False
    except ValueError:
        return False

    basestring = f"v0:{timestamp}:{body.decode('utf-8')}".encode("utf-8")
    expected = "v0=" + hmac.new(
        signing_secret.encode("utf-8"), basestring, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


def build_reason_modal(private_metadata: str) -> dict:
    """
    Build the 'reject reason' modal view.

    The reason input is ``optional`` — the approver can submit with the field
    left blank to reject without providing a reason.
    """
    return {
        "type": "modal",
        "callback_id": "reject_reason_modal",
        "private_metadata": private_metadata,
        "title":  {"type": "plain_text", "text": "Reject ERP Update"},
        "submit": {"type": "plain_text", "text": "Submit Rejection"},
        "close":  {"type": "plain_text", "text": "Cancel"},
        "blocks": [
            {
                "type": "input",
                "optional": True,                       # ← blank submit allowed
                "block_id": "reason_block",
                "label": {"type": "plain_text", "text": "Reason for rejection (optional)"},
                "element": {
                    "type": "plain_text_input",
                    "action_id": "reason_input",
                    "multiline": True,
                    "placeholder": {
                        "type": "plain_text",
                        "text": "Leave blank to reject without a reason, then click Submit.",
                    },
                },
            }
        ],
    }


async def open_reason_modal(trigger_id: str, private_metadata: str) -> bool:
    """Open the rejection-reason modal via Slack views.open (needs bot token)."""
    token = os.getenv("SLACK_BOT_TOKEN", "")
    if not token:
        logger.error("[slack] SLACK_BOT_TOKEN not set — cannot open modal.")
        return False

    payload = {"trigger_id": trigger_id, "view": build_reason_modal(private_metadata)}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                "https://slack.com/api/views.open",
                headers={"Authorization": f"Bearer {token}"},
                json=payload,
            )
        data = resp.json()
        if not data.get("ok"):
            logger.error("[slack] views.open failed: %s", data.get("error"))
            return False
        return True
    except Exception as e:
        logger.error("[slack] views.open request failed: %s", e)
        return False


async def update_message_via_response_url(response_url: str, text: str) -> bool:
    """Replace the original approval message (removing the buttons) via response_url."""
    if not response_url:
        return False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(
                response_url,
                json={"replace_original": True, "text": text},
            )
        return True
    except Exception as e:
        logger.error("[slack] response_url update failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# 발송 함수 (비동기 — FastAPI 핸들러에서 사용)
# ---------------------------------------------------------------------------

async def send_approval_request_async(
    action: dict,
    thread_id: str,
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

    message = _build_message(action, thread_id)
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(webhook_url, json=message)
            resp.raise_for_status()
            logger.info("[slack] Approval request sent (thread_id=%s)", thread_id)
            return True
    except Exception as e:
        logger.error("[slack] Failed to send message: %s", e)
        return False
