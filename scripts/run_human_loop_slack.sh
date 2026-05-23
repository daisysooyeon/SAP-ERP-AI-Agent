#!/usr/bin/env bash
# =============================================================================
# scripts/run_human_loop_slack.sh
# Human-in-the-Loop + Slack 알림 end-to-end 테스트
#
# 수행 단계:
#   1. FastAPI 서버를 백그라운드로 실행
#   2. 서버가 준비될 때까지 대기 (/api/health 폴링)
#   3. 실제 DB 주문(#0000015353)에 대한 취소 요청 이메일로 /api/run 호출
#   4. Slack 알림 전송 여부 확인 및 응답 출력
#   5. 사용자가 Enter를 누르면 서버 종료
#
# 테스트 케이스 출처:
#   data/eval/router_test_cases_gen.json — ACTION_ONLY (CANCEL_ITEM)
#   사용 주문: 0000015353, item 10 (DB 검증 완료)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Python / uvicorn 경로 ─────────────────────────────────────────────────────
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

# ── 색상 ──────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_step()  { echo -e "${CYAN}[STEP]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ── 배너 ──────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   SAP ERP AI Agent — Human-in-the-Loop + Slack 테스트 ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── 테스트 이메일 정의 ────────────────────────────────────────────────────────
# 출처: data/eval/router_test_cases_gen.json > ACTION_ONLY (CANCEL_ITEM)
# 주문 0000015353, item 10 — DB에서 존재가 검증된 실제 주문
TEST_EMAIL="Dear SAP team,

Please cancel item 10 of sales order 0000015353. The customer has withdrawn their purchase request and we need to process this cancellation immediately.

Best regards,
Daisy"

# ── 서버 시작 ─────────────────────────────────────────────────────────────────
log_step "1/4  FastAPI 서버를 백그라운드로 시작합니다..."

cd "${PROJECT_ROOT}"
LOG_FILE="/tmp/sap_server_$$.log"

"${UVICORN}" src.api.server:app \
  --host 0.0.0.0 \
  --port 8000 \
  > "${LOG_FILE}" 2>&1 &

SERVER_PID=$!
log_info "서버 PID: ${SERVER_PID}  (로그: ${LOG_FILE})"

# 서버 종료 시 PID 정리
cleanup() {
  echo ""
  log_info "서버를 종료합니다 (PID: ${SERVER_PID})..."
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
  log_info "서버 종료 완료."
}
trap cleanup EXIT

# ── 서버 준비 대기 ────────────────────────────────────────────────────────────
log_step "2/4  서버 준비 대기 중..."

MAX_WAIT=90
ELAPSED=0
until "${PYTHON}" -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health', timeout=2)" > /dev/null 2>&1; do
  if [[ ${ELAPSED} -ge ${MAX_WAIT} ]]; then
    log_error "서버가 ${MAX_WAIT}초 내에 시작되지 않았습니다."
    log_warn "서버 로그:"
    tail -20 "${LOG_FILE}"
    exit 1
  fi
  sleep 1
  ELAPSED=$((ELAPSED + 1))
  printf "."
done
echo ""
log_info "서버 준비 완료 (${ELAPSED}초 소요)"

# 공개 URL 확인 (ngrok 사용 시)
PUBLIC_URL=$("${PYTHON}" -c "
import urllib.request, json
r = urllib.request.urlopen('http://localhost:8000/api/health', timeout=5)
d = json.loads(r.read())
print(d.get('public_url', 'http://localhost:8000'))
" 2>/dev/null || echo "http://localhost:8000")
log_info "Public URL: ${PUBLIC_URL}"

# ── /api/run 호출 ─────────────────────────────────────────────────────────────
log_step "3/4  테스트 이메일로 /api/run 호출..."
echo ""
echo "  이메일 내용:"
echo "${TEST_EMAIL}" | sed 's/^/    /'
echo ""

RESPONSE=$("${PYTHON}" -c "
import urllib.request, json, sys
email = sys.stdin.read()
payload = json.dumps({'email_text': email}).encode()
req = urllib.request.Request(
    'http://localhost:8000/api/run',
    data=payload,
    headers={'Content-Type': 'application/json'},
    method='POST',
)
r = urllib.request.urlopen(req, timeout=300)
print(r.read().decode())
" <<< "${TEST_EMAIL}") || {
    log_error "/api/run 호출 실패"
    log_warn "서버 로그 마지막 30줄:"
    tail -30 "${LOG_FILE}"
    exit 1
  }

echo ""
log_info "/api/run 응답:"
echo "${RESPONSE}" | "${PYTHON}" -c "import sys,json; d=json.load(sys.stdin); [print(f'  {k}: {v}') for k,v in d.items()]" 2>/dev/null || echo "${RESPONSE}"

# ── 결과 분석 ─────────────────────────────────────────────────────────────────
echo ""
REQUIRES_APPROVAL=$(echo "${RESPONSE}" | "${PYTHON}" -c "import sys,json; print(json.load(sys.stdin).get('requires_approval', False))" 2>/dev/null || echo "False")
THREAD_ID=$(echo "${RESPONSE}" | "${PYTHON}" -c "import sys,json; print(json.load(sys.stdin).get('thread_id',''))" 2>/dev/null || echo "")

if [[ "${REQUIRES_APPROVAL}" == "True" ]]; then
  log_info "✅ 승인 대기 중 — Slack 알림이 전송되었는지 확인하세요."
  echo ""
  echo "  Thread ID : ${THREAD_ID}"
  echo ""
  echo "  수동 승인/거부 URL:"
  echo "    ✅ 승인: ${PUBLIC_URL}/api/approve?thread_id=${THREAD_ID}&approved=true"
  echo "    ❌ 거부: ${PUBLIC_URL}/api/approve?thread_id=${THREAD_ID}&approved=false"
  echo ""
  log_step "4/4  Slack 버튼을 클릭하거나 위 URL로 승인/거부 후 Enter를 눌러 서버를 종료하세요."
else
  log_warn "승인 요청이 트리거되지 않았습니다."
  echo "  erp_status: $(echo "${RESPONSE}" | "${PYTHON}" -c "import sys,json; print(json.load(sys.stdin).get('erp_status','N/A'))" 2>/dev/null)"
  echo ""
  log_warn "서버 로그 마지막 20줄:"
  tail -20 "${LOG_FILE}"
  echo ""
  log_step "4/4  Enter를 눌러 서버를 종료하세요."
fi

echo ""
read -r -p "  [Enter] 서버 종료 > "
