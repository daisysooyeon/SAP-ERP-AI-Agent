"""
src/slack/notifier.py
Slack Incoming Webhook을 통한 ERP 업데이트 승인 요청 발송
"""

import os
import httpx

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")


def send_approval_request(action: dict, thread_id: str, server_base_url: str = "http://localhost:8000") -> None:
    """
    Slack에 ERP 업데이트 승인 요청 메시지 발송
    
    Args:
        action: ERPAction dict (order_id, item_no, field, new_value, reason)
        thread_id: LangGraph 스레드 ID
        server_base_url: FastAPI 서버 주소 (Slack에서 접근 가능해야 함)
    """
    approve_url = f"{server_base_url}/api/approve?thread_id={thread_id}&approved=true"
    reject_url = f"{server_base_url}/api/approve?thread_id={thread_id}&approved=false"

    message = {
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*🔔 ERP 업데이트 승인 요청*\n"
                        f"• 오더번호: `{action.get('order_id', '-')}`\n"
                        f"• 아이템: `{action.get('item_no', '-')}`\n"
                        f"• 변경 내용: {action.get('field', '-')} → `{action.get('new_value', '-')}`\n"
                        f"• 사유: {action.get('reason', '-')}"
                    ),
                },
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "✅ 승인"},
                        "style": "primary",
                        "url": approve_url,
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "❌ 거절"},
                        "style": "danger",
                        "url": reject_url,
                    },
                ],
            },
        ]
    }

    if not SLACK_WEBHOOK_URL:
        print("[WARN] SLACK_WEBHOOK_URL이 설정되지 않았습니다. 메시지를 전송하지 않습니다.")
        return

    response = httpx.post(SLACK_WEBHOOK_URL, json=message, timeout=5.0)
    response.raise_for_status()
    print(f"[OK] Slack 승인 요청 발송 완료 (thread_id={thread_id})")
