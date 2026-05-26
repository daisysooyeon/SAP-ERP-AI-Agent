#!/usr/bin/env bash
# =============================================================================
# scripts/run_eval_e2e.sh
# End-to-End evaluation — LLM-as-a-Judge (synthesizer quality)
#
# Usage:
#   bash scripts/run_eval_e2e.sh                          # all 90 cases
#   bash scripts/run_eval_e2e.sh --dry-run                # first 3 only
#   bash scripts/run_eval_e2e.sh --report reports/x.json  # custom report path
#
# What it measures:
#   - faithfulness  (1-5): No hallucinated facts
#   - correctness   (1-5): ERP/RAG results accurately reflected
#   - format        (1-5): Proper business email structure
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPORT_DIR="${PROJECT_ROOT}/reports"
DEFAULT_REPORT="${REPORT_DIR}/e2e_eval.json"
TEST_CASES="${PROJECT_ROOT}/data/eval/router_test_cases_gen.json"

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
echo "║     SAP ERP AI Agent — E2E Evaluation                ║"
echo "║     LLM-as-a-Judge · Faithfulness / Correctness /    ║"
echo "║     Format                                           ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Pre-flight checks ─────────────────────────────────────────────────────────
log_info "Project root : ${PROJECT_ROOT}"

if [[ ! -f "${TEST_CASES}" ]]; then
  log_error "Test cases not found: ${TEST_CASES}"
  exit 1
fi
log_info "Test cases   : ${TEST_CASES}"

if [[ ! -f "${PROJECT_ROOT}/.env" ]]; then
  log_warn ".env not found — OPENROUTER_API_KEY may not be set."
else
  log_info ".env loaded"
fi

# ── --report default injection ─────────────────────────────────────────────────
EXTRA_ARGS=("$@")
HAS_REPORT=false
for arg in "${EXTRA_ARGS[@]:-}"; do
  [[ "$arg" == "--report" ]] && HAS_REPORT=true
done

if [[ "${HAS_REPORT}" == false ]]; then
  mkdir -p "${REPORT_DIR}"
  EXTRA_ARGS+=("--report" "${DEFAULT_REPORT}")
  log_info "Report will be saved to: ${DEFAULT_REPORT}"
fi

echo ""

# ── Run ───────────────────────────────────────────────────────────────────────
cd "${PROJECT_ROOT}"
export PYTHONPATH="${WIN_PROJECT_ROOT}:${PYTHONPATH:-}"

WIN_TEST_CASES="$(wslpath -w "${TEST_CASES}" 2>/dev/null || echo "${TEST_CASES}")"

# Convert --report path to Windows path if needed
WIN_EXTRA_ARGS=()
for arg in "${EXTRA_ARGS[@]}"; do
  if [[ -n "${_NEXT_IS_PATH:-}" ]]; then
    WIN_EXTRA_ARGS+=("$(wslpath -w "${arg}" 2>/dev/null || echo "${arg}")")
    _NEXT_IS_PATH=""
  elif [[ "$arg" == "--report" ]]; then
    WIN_EXTRA_ARGS+=("${arg}")
    _NEXT_IS_PATH=1
  else
    WIN_EXTRA_ARGS+=("${arg}")
  fi
done

"${PYTHON}" -m src.evaluation.eval_e2e \
  --test-cases "${WIN_TEST_CASES}" \
  "${WIN_EXTRA_ARGS[@]}"

EXIT_CODE=$?

echo ""
if [[ ${EXIT_CODE} -eq 0 ]]; then
  log_info "✅  E2E evaluation completed successfully."
else
  log_error "❌  E2E evaluation failed — check output above."
fi

exit ${EXIT_CODE}
