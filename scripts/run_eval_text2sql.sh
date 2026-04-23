#!/usr/bin/env bash
# =============================================================================
# scripts/run_eval_text2sql.sh
# Run the Text-to-SQL accuracy evaluation (Phase 3)
#
# Usage:
#   bash scripts/run_eval_text2sql.sh                       # default settings
#   bash scripts/run_eval_text2sql.sh --report reports/x.json # save report to file
#
# All extra arguments are forwarded to eval_text2sql.py.
# =============================================================================

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPORT_DIR="${PROJECT_ROOT}/reports"
DEFAULT_REPORT="${REPORT_DIR}/text2sql_eval_$(date +%Y%m%d_%H%M%S).json"
TEST_CASES="${PROJECT_ROOT}/data/eval/text2sql_test_cases.json"
DB_PATH="${PROJECT_ROOT}/data/sap_erp.db"

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── Banner ───────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║      SAP ERP AI Agent — Text-to-SQL Eval         ║"
echo "║      Phase 3 · SQLite DB Validation Queries      ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── Pre-flight checks ────────────────────────────────────────────────────────
log_info "Project root : ${PROJECT_ROOT}"

# 1. Test cases file
if [[ ! -f "${TEST_CASES}" ]]; then
  log_error "Test cases file not found: ${TEST_CASES}"
  log_error "Make sure data/eval/text2sql_test_cases.json exists."
  exit 1
fi
log_info "Test cases   : ${TEST_CASES}"

# 2. SQLite DB file
if [[ ! -f "${DB_PATH}" ]]; then
  log_error "SQLite DB not found: ${DB_PATH}"
  log_error "Run 'python -m src.db.setup_sqlite' to create the database."
  exit 1
fi
log_info "Database     : ${DB_PATH}"

# ── Resolve --report argument (inject default if caller didn't pass one) ─────
EXTRA_ARGS=("$@")
HAS_REPORT=false
for arg in "${EXTRA_ARGS[@]}"; do
  [[ "$arg" == "--report" ]] && HAS_REPORT=true
done

if [[ "${HAS_REPORT}" == false ]]; then
  mkdir -p "${REPORT_DIR}"
  EXTRA_ARGS+=("--report" "${DEFAULT_REPORT}")
  log_info "Report will be saved to: ${DEFAULT_REPORT}"
fi

echo ""

# ── Run evaluation ───────────────────────────────────────────────────────────
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python -m src.evaluation.eval_text2sql \
  --test-cases "${TEST_CASES}" \
  --db "${DB_PATH}" \
  "${EXTRA_ARGS[@]}"

EXIT_CODE=$?

echo ""
if [[ ${EXIT_CODE} -eq 0 ]]; then
  log_info "✅  Evaluation completed successfully."
else
  log_warn "❌  Evaluation FAILED or encountered errors — check the report for details."
fi

exit ${EXIT_CODE}
