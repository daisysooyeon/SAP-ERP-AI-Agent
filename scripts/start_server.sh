#!/usr/bin/env bash
# =============================================================================
# scripts/start_server.sh
# SAP ERP AI Agent FastAPI 서버 실행
#
# Usage:
#   bash scripts/start_server.sh           # 기본 (localhost:8000)
#   bash scripts/start_server.sh --reload  # 코드 변경 자동 감지 (개발용)
#
# ngrok 자동 터널 설정:
#   .env에 NGROK_AUTHTOKEN을 추가하면 서버 시작 시 ngrok 터널이 자동으로 열립니다.
#   ngrok 계정: https://dashboard.ngrok.com/get-started/your-authtoken
#
# Vercel 배포 시:
#   SERVER_BASE_URL=https://your-project.vercel.app 환경변수만 변경하면 됩니다.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Python interpreter ────────────────────────────────────────────────────────
if [[ -f "${PROJECT_ROOT}/sap/Scripts/python.exe" ]]; then
  PYTHON="${PROJECT_ROOT}/sap/Scripts/python.exe"
  UVICORN="${PROJECT_ROOT}/sap/Scripts/uvicorn.exe"
elif command -v python3 &>/dev/null; then
  PYTHON="python3"
  UVICORN="uvicorn"
else
  PYTHON="python"
  UVICORN="uvicorn"
fi

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log_info() { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║     SAP ERP AI Agent — FastAPI Server                ║"
echo "║     Human-in-the-Loop + Slack 연동                   ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── .env 확인 ─────────────────────────────────────────────────────────────────
ENV_FILE="${PROJECT_ROOT}/.env"
if [[ -f "${ENV_FILE}" ]]; then
  log_info ".env 파일 로드됨"
  # NGROK_AUTHTOKEN 설정 여부 안내
  if grep -q "NGROK_AUTHTOKEN" "${ENV_FILE}" 2>/dev/null; then
    log_info "NGROK_AUTHTOKEN 감지됨 — 서버 시작 시 ngrok 터널이 자동으로 열립니다."
  else
    log_warn "NGROK_AUTHTOKEN 미설정 — localhost:8000으로만 접근 가능합니다."
    log_warn "외부 접근이 필요하면 .env에 NGROK_AUTHTOKEN=<your_token> 추가하세요."
    log_warn "토큰 발급: https://dashboard.ngrok.com/get-started/your-authtoken"
  fi

  if ! grep -q "SLACK_WEBHOOK_URL" "${ENV_FILE}" 2>/dev/null; then
    log_warn "SLACK_WEBHOOK_URL 미설정 — Slack 알림이 전송되지 않습니다."
  else
    log_info "SLACK_WEBHOOK_URL 감지됨 — 승인 요청 시 Slack DM이 발송됩니다."
  fi
else
  log_warn ".env 파일이 없습니다. API 키 설정이 필요합니다."
fi

echo ""
log_info "서버 시작: http://localhost:8000"
log_info "Swagger UI: http://localhost:8000/docs"
echo ""

# ── 서버 실행 ─────────────────────────────────────────────────────────────────
cd "${PROJECT_ROOT}"

RELOAD_FLAG=""
for arg in "$@"; do
  [[ "$arg" == "--reload" ]] && RELOAD_FLAG="--reload"
done

"${UVICORN}" src.api.server:app \
  --host 0.0.0.0 \
  --port 8000 \
  ${RELOAD_FLAG}
