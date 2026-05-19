#!/usr/bin/env bash
# =============================================================================
# scripts/run_eval_worker_b.sh
# Run the Worker B RAG pipeline evaluation
#   (Hit Rate · NDCG@3 · MRR · Context Recall · Faithfulness)
#
# Usage:
#   bash scripts/run_eval_worker_b.sh                          # full eval (all cases)
#   bash scripts/run_eval_worker_b.sh --skip-llm-judge         # retrieval metrics only (fast)
#   bash scripts/run_eval_worker_b.sh --report reports/x.json  # custom report path
#
# All extra arguments are forwarded to eval_worker_b.py.
# =============================================================================

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPORT_DIR="${PROJECT_ROOT}/reports"
DEFAULT_REPORT="${REPORT_DIR}/worker_b_eval.json"
TEST_CASES="${PROJECT_ROOT}/data/eval/rag_test_cases.json"
CHROMA_DB="${PROJECT_ROOT}/chroma_db"

# ── Python interpreter (Windows venv 우선, 없으면 시스템 python) ───────────────
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
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║      SAP ERP AI Agent — Worker B RAG Eval        ║"
echo "║  Hit Rate · NDCG@3 · MRR · Ctx Recall · Faith   ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── Pre-flight checks ─────────────────────────────────────────────────────────
log_info "Project root : ${PROJECT_ROOT}"

# 1. Test cases file
if [[ ! -f "${TEST_CASES}" ]]; then
  log_error "Test cases file not found: ${TEST_CASES}"
  log_error "Run 'python -m src.data.generate_rag_dataset' to generate it."
  exit 1
fi
log_info "Test cases   : ${TEST_CASES}"

# 2. ChromaDB directory
if [[ ! -d "${CHROMA_DB}" ]]; then
  log_error "ChromaDB not found: ${CHROMA_DB}"
  log_error "Run 'python -m src.rag.ingest' to build the vector store."
  exit 1
fi
log_info "ChromaDB     : ${CHROMA_DB}"

# 3. .env file check
if [[ ! -f "${PROJECT_ROOT}/.env" ]]; then
  log_warn ".env file not found — OPENROUTER_API_KEY may not be set."
fi

# ── Resolve --report argument (inject default if caller didn't pass one) ──────
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

# ── Run evaluation ────────────────────────────────────────────────────────────
cd "${PROJECT_ROOT}"
export PYTHONPATH="${WIN_PROJECT_ROOT}:${PYTHONPATH:-}"

# --report 인자도 Windows 경로로 변환
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

"${PYTHON}" -m src.evaluation.eval_worker_b \
  "${WIN_EXTRA_ARGS[@]}"

EXIT_CODE=$?

echo ""
if [[ ${EXIT_CODE} -eq 0 ]]; then
  log_info "✅  Evaluation completed successfully."
else
  log_warn "❌  Evaluation FAILED or encountered errors — check the report for details."
fi

exit ${EXIT_CODE}
