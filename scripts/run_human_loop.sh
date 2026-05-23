#!/usr/bin/env bash
# =============================================================================
# scripts/run_human_loop.sh
# Human-in-the-Loop end-to-end test (no server or Slack required)
#
# Usage:
#   bash scripts/run_human_loop.sh                    # both approve + reject
#   bash scripts/run_human_loop.sh --approve          # approve scenario only
#   bash scripts/run_human_loop.sh --reject           # reject scenario only
#   bash scripts/run_human_loop.sh --email "Cancel order 4976 item 10"
#
# What it tests:
#   1. Agent processes an ERP email → graph pauses at PENDING_APPROVAL
#   2. Approval scenario  → SQLite UPDATE → erp_action_status=SUCCESS
#   3. Rejection scenario → no DB change  → erp_action_status=REJECTED
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Python interpreter ────────────────────────────────────────────────────────
if [[ -f "${PROJECT_ROOT}/sap/Scripts/python.exe" ]]; then
  PYTHON="${PROJECT_ROOT}/sap/Scripts/python.exe"
  WIN_PROJECT_ROOT="$(wslpath -w "${PROJECT_ROOT}")"
elif command -v python3 &>/dev/null; then
  PYTHON="python3"
  WIN_PROJECT_ROOT="${PROJECT_ROOT}"
else
  PYTHON="python"
  WIN_PROJECT_ROOT="${PROJECT_ROOT}"
fi

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║     SAP ERP AI Agent — Human-in-the-Loop Test        ║"
echo "║     No server or Slack required                       ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Pre-flight checks ─────────────────────────────────────────────────────────
log_info "Project root : ${PROJECT_ROOT}"

DB_PATH="${PROJECT_ROOT}/data/sap_erp.db"
if [[ ! -f "${DB_PATH}" ]]; then
  log_error "SQLite DB not found: ${DB_PATH}"
  exit 1
fi
log_info "Database     : ${DB_PATH}"

if [[ ! -f "${PROJECT_ROOT}/.env" ]]; then
  log_warn ".env not found — OPENROUTER_API_KEY may not be set."
else
  log_info ".env loaded"
fi

echo ""

# ── Run test ──────────────────────────────────────────────────────────────────
cd "${PROJECT_ROOT}"
export PYTHONPATH="${WIN_PROJECT_ROOT}:${PYTHONPATH:-}"

"${PYTHON}" -m src.evaluation.test_human_loop "$@"

EXIT_CODE=$?

echo ""
if [[ ${EXIT_CODE} -eq 0 ]]; then
  log_info "✅  All scenarios passed."
else
  log_error "❌  One or more scenarios failed — check output above."
fi

exit ${EXIT_CODE}
