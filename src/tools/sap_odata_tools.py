"""
src/tools/sap_odata_tools.py
SAP CE_SALESORDER_0001 (OData v4 A2X) 샌드박스 API 호출 Tool
"""

import os
import httpx
from src.api.schemas import ERPActionRequest

SAP_SANDBOX_BASE = (
    "https://sandbox.api.sap.com/s4hanacloud/sap/opu/odata4/sap"
    "/api_salesorder/srvd_a2x/sap/salesorder/0001"
)


async def call_sap_odata_patch(action: ERPActionRequest) -> dict:
    """
    SAP Business Accelerator Hub 샌드박스 OData v4 PATCH 호출
    실제 데이터를 수정하지 않고, 성공 응답(200/204)만 확인
    
    엔드포인트: PATCH /SalesOrderItem(SalesOrder='{id}',SalesOrderItem='{item}')
    """
    sap_api_key = os.getenv("SAP_API_KEY", "")
    url = (
        f"{SAP_SANDBOX_BASE}/SalesOrderItem"
        f"(SalesOrder='{action.order_id}',SalesOrderItem='{action.item_no}')"
    )

    payload: dict = {}
    if action.action_type == "CHANGE_QTY" and action.new_quantity is not None:
        payload["RequestedQuantity"] = str(action.new_quantity)
    elif action.action_type == "CHANGE_DATE" and action.new_date is not None:
        payload["RequestedDeliveryDate"] = action.new_date  # YYYY-MM-DD

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.patch(
            url,
            json=payload,
            headers={
                "APIKey": sap_api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

    return {"status_code": response.status_code, "body": response.text}
