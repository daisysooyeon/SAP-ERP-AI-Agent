#!/usr/bin/env bash
# =============================================================================
# scripts/run_eval_router.sh
# Run the router accuracy evaluation (Phase 2 final deliverable)
#
# Usage:
#   bash scripts/run_eval_router.sh                      # default settings
#   bash scripts/run_eval_router.sh --quiet              # suppress per-sample output
#   bash scripts/run_eval_router.sh --out reports/x.json # save report to file
#
# All extra arguments are forwarded to eval_router.py.
# =============================================================================

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPORT_DIR="${PROJECT_ROOT}/reports"
DEFAULT_REPORT="${REPORT_DIR}/router_eval_result.json"
TEST_CASES="${TEST_CASES:-${PROJECT_ROOT}/data/eval/router_test_cases_gen.json}"

# ── Python interpreter (Windows venv 우선, 없으면 시스템 python) ──────────────
if [[ -f "${PROJECT_ROOT}/sap/Scripts/python.exe" ]]; then
  PYTHON="${PROJECT_ROOT}/sap/Scripts/python.exe"
  # Windows python.exe는 WSL 경로(/mnt/c/...)를 이해하지 못하므로 Windows 경로로 변환
  WIN_PROJECT_ROOT="$(wslpath -w "${PROJECT_ROOT}")"
elif command -v python3 &>/dev/null; then
  PYTHON="python3"
  WIN_PROJECT_ROOT="${PROJECT_ROOT}"
else
  PYTHON="python"
  WIN_PROJECT_ROOT="${PROJECT_ROOT}"
fi

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── Banner ───────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║      SAP ERP AI Agent — Router Evaluation        ║"
echo "║      Phase 2 · Intent Classification Report      ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── Pre-flight checks ────────────────────────────────────────────────────────
log_info "Project root : ${PROJECT_ROOT}"

# 1. Test cases file
if [[ ! -f "${TEST_CASES}" ]]; then
  log_error "Test cases file not found: ${TEST_CASES}"
  log_error "Make sure data/eval/router_test_cases.json exists."
  exit 1
fi
log_info "Test cases   : ${TEST_CASES}"

# 2. Read router model name from configs.yaml (env var override still works)
if [[ -z "${ROUTER_MODEL:-}" ]]; then
  ROUTER_MODEL=$("${PYTHON}" -c "
import sys; sys.path.insert(0, '${WIN_PROJECT_ROOT}')
from src.config import get_config
print(get_config().models.router.name)
" 2>/dev/null) || ROUTER_MODEL="qwen/qwen3-8b"
fi

# 2.5 Read router provider from configs.yaml
if [[ -z "${ROUTER_PROVIDER:-}" ]]; then
  ROUTER_PROVIDER=$("${PYTHON}" -c "
import sys; sys.path.insert(0, '${WIN_PROJECT_ROOT}')
from src.config import get_config
print(get_config().models.router.provider)
" 2>/dev/null) || ROUTER_PROVIDER="openrouter"
fi

# 3. Ollama running?
if [[ "${ROUTER_PROVIDER}" == "ollama" ]]; then
  if ! curl -sf http://localhost:11434 > /dev/null 2>&1; then
    log_error "Ollama is not reachable at http://localhost:11434"
    log_error "Start Ollama serve or switch provider in configs.yaml"
    exit 1
  fi
  log_info "Ollama       : ✓ reachable"

  # 4. Model available?
  if ! ollama list 2>/dev/null | grep -q "${ROUTER_MODEL}"; then
    log_warn "Model '${ROUTER_MODEL}' not found locally — pulling now …"
    ollama pull "${ROUTER_MODEL}"
  fi
  log_info "Model        : ${ROUTER_MODEL}"
else
  log_info "Provider     : ${ROUTER_PROVIDER} (skipping Ollama checks)"
  log_info "Model        : ${ROUTER_MODEL}"
fi

# ── Resolve --out argument (inject default if caller didn't pass one) ─────────
EXTRA_ARGS=("$@")
HAS_OUT=false
for arg in "${EXTRA_ARGS[@]}"; do
  [[ "$arg" == "--out" ]] && HAS_OUT=true
done

if [[ "${HAS_OUT}" == false ]]; then
  mkdir -p "${REPORT_DIR}"
  EXTRA_ARGS+=("--out" "${DEFAULT_REPORT}")
  log_info "Report will be saved to: ${DEFAULT_REPORT}"
fi

echo ""

# ── Run evaluation ───────────────────────────────────────────────────────────
cd "${PROJECT_ROOT}"
export PYTHONPATH="${WIN_PROJECT_ROOT}:${PYTHONPATH:-}"
export ROUTER_MODEL

# Windows python.exe용 경로 변환
WIN_TEST_CASES="$(wslpath -w "${TEST_CASES}" 2>/dev/null || echo "${TEST_CASES}")"

# --out 인자도 Windows 경로로 변환
WIN_EXTRA_ARGS=()
for arg in "${EXTRA_ARGS[@]}"; do
  if [[ -n "${_NEXT_IS_PATH:-}" ]]; then
    WIN_EXTRA_ARGS+=("$(wslpath -w "${arg}" 2>/dev/null || echo "${arg}")")
    _NEXT_IS_PATH=""
  elif [[ "$arg" == "--out" ]]; then
    WIN_EXTRA_ARGS+=("${arg}")
    _NEXT_IS_PATH=1
  else
    WIN_EXTRA_ARGS+=("${arg}")
  fi
done

"${PYTHON}" -m src.evaluation.eval_router \
  --test-cases "${WIN_TEST_CASES}" \
  "${WIN_EXTRA_ARGS[@]}"

EXIT_CODE=$?

echo ""
if [[ ${EXIT_CODE} -eq 0 ]]; then
  log_info "✅  Evaluation PASSED (accuracy ≥ 99%)"
else
  log_warn "❌  Evaluation FAILED (accuracy < 99%) — check the report for details."
fi

exit ${EXIT_CODE}
