#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run RAG from fixed analysis output.

In my mode this copies:
  <analysis-repo>/artifacts/final/final_scripts/output/rag_index/rag_documents.jsonl

to:
  data/inbox/control_flow_rag_documents.jsonl

In combined mode this copies:
  <analysis-repo>/artifacts/final/final_scripts/output/combined/rag_index/<PROGRAM>_combined.jsonl

to:
  data/inbox/control_flow_rag_documents_combined.jsonl

Each mode uses a separate Chroma dir and collection, so tests stay independent.

Usage:
  ./scripts/run_fixed_input_rag.sh --program PDHASI06 --analysis-repo ../legacy-program-analysis
  ./scripts/run_fixed_input_rag.sh --mode combined --program PDHASI06 --analysis-repo ../legacy-program-analysis

Options:
  --mode my|combined    Test mode. Default: my.
  --program NAME         Program to use for direct final_scripts answers.
  --analysis-repo PATH   Analysis repo path. Default: ../legacy-program-analysis, then ../control_flow.
  --build-analysis       Run the fixed-input analysis pipeline before indexing.
  --port N               UI port. Default: 8000.
  --no-install           Do not create/use venv and pip install -e .
  --no-server            Only copy/inspect/sync; do not start uvicorn.
  --pull-models          Pull Ollama models before indexing.
  -h, --help             Show this help.
EOF
}

PROGRAM=""
MODE="my"
ANALYSIS_REPO=""
PORT="8000"
INSTALL=1
START_SERVER=1
PULL_MODELS=0
BUILD_ANALYSIS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --program)
      PROGRAM="${2:-}"
      shift 2
      ;;
    --analysis-repo)
      ANALYSIS_REPO="${2:-}"
      shift 2
      ;;
    --build-analysis)
      BUILD_ANALYSIS=1
      shift
      ;;
    --port)
      PORT="${2:-}"
      shift 2
      ;;
    --no-install)
      INSTALL=0
      shift
      ;;
    --no-server)
      START_SERVER=0
      shift
      ;;
    --pull-models)
      PULL_MODELS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$PROGRAM" ]]; then
  echo "Missing --program." >&2
  usage
  exit 2
fi

case "$MODE" in
  my|combined) ;;
  *)
    echo "--mode must be one of: my, combined" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "$ANALYSIS_REPO" ]]; then
  if [[ -d "../legacy-program-analysis" ]]; then
    ANALYSIS_REPO="../legacy-program-analysis"
  elif [[ -d "../control_flow" ]]; then
    ANALYSIS_REPO="../control_flow"
  else
    ANALYSIS_REPO="../legacy-program-analysis"
  fi
fi

resolve_path() {
  local path="$1"
  python - "$path" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
}

PROGRAM_UPPER="$(printf '%s' "$PROGRAM" | tr '[:lower:]' '[:upper:]')"
PROGRAM_LOWER="$(printf '%s' "$PROGRAM" | tr '[:upper:]' '[:lower:]')"
ANALYSIS_REPO_ABS="$(resolve_path "$ANALYSIS_REPO")"

PYTHON_BIN="python3"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

if [[ "$MODE" == "combined" ]]; then
  SOURCE_JSONL="$ANALYSIS_REPO_ABS/artifacts/final/final_scripts/output/combined/rag_index/${PROGRAM_UPPER}_combined.jsonl"
  FINAL_SCRIPTS="$ANALYSIS_REPO_ABS/artifacts/final/final_scripts/output/combined/final_scripts/$PROGRAM_UPPER"
  TARGET_JSONL="data/inbox/control_flow_rag_documents_combined.jsonl"
  COLLECTION="cobol-combined-${PROGRAM_LOWER}"
  CHROMA_DIR="data/chroma-combined-${PROGRAM_LOWER}"
  RUN_HINT="python scripts/pipeline/run_fixed_input.py --program $PROGRAM_UPPER --mode both"
else
  SOURCE_JSONL="$ANALYSIS_REPO_ABS/artifacts/final/final_scripts/output/rag_index/rag_documents.jsonl"
  FINAL_SCRIPTS="$ANALYSIS_REPO_ABS/artifacts/final/final_scripts/output/program_artifacts/programs/$PROGRAM_UPPER/artifacts"
  TARGET_JSONL="data/inbox/control_flow_rag_documents.jsonl"
  COLLECTION="cobol-fixed-${PROGRAM_LOWER}"
  CHROMA_DIR="data/chroma-fixed-${PROGRAM_LOWER}"
  RUN_HINT="python scripts/pipeline/run_fixed_input.py --program $PROGRAM_UPPER --mode my"
fi

if [[ "$BUILD_ANALYSIS" -eq 1 ]]; then
  ANALYSIS_MODE="my"
  if [[ "$MODE" == "combined" ]]; then
    ANALYSIS_MODE="both"
  fi
  echo
  echo "Building analysis output: $ANALYSIS_MODE"
  (
    cd "$ANALYSIS_REPO_ABS"
    "$PYTHON_BIN" scripts/pipeline/run_fixed_input.py --program "$PROGRAM_UPPER" --mode "$ANALYSIS_MODE"
  )
fi

if [[ ! -f "$SOURCE_JSONL" ]]; then
  echo "Generated RAG JSONL not found: $SOURCE_JSONL" >&2
  echo "Run this in the analysis repo first:" >&2
  echo "  $RUN_HINT" >&2
  exit 1
fi

if [[ ! -d "$FINAL_SCRIPTS" ]]; then
  echo "Final scripts artifact root not found: $FINAL_SCRIPTS" >&2
  exit 1
fi

mkdir -p data/inbox data/archive
if [[ -f "$TARGET_JSONL" ]]; then
  STAMP="$(date +%Y%m%d%H%M%S)"
  ARCHIVE_NAME="$(basename "$TARGET_JSONL" .jsonl)"
  cp "$TARGET_JSONL" "data/archive/${ARCHIVE_NAME}.${STAMP}.jsonl"
fi
cp "$SOURCE_JSONL" "$TARGET_JSONL"

if [[ "$INSTALL" -eq 1 ]]; then
  if [[ ! -d ".venv" ]]; then
    "$PYTHON_BIN" -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install --upgrade pip
  pip install -e .
elif [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if [[ "$PULL_MODELS" -eq 1 ]]; then
  ollama pull granite-code:8b-instruct
  ollama pull mxbai-embed-large:latest
fi

export COBOL_RAG_FINAL_SCRIPTS_DIR="$FINAL_SCRIPTS"
export COBOL_RAG_COLLECTION="$COLLECTION"
export COBOL_RAG_CHROMA_DIR="$CHROMA_DIR"
export COBOL_RAG_LLM_POLISH_FINAL_SCRIPTS="${COBOL_RAG_LLM_POLISH_FINAL_SCRIPTS:-false}"

echo
echo "Mode: $MODE"
echo "Program: $PROGRAM_UPPER"
echo "Copied: $SOURCE_JSONL"
echo "To: $TARGET_JSONL"
echo "Final scripts: $COBOL_RAG_FINAL_SCRIPTS_DIR"
echo "Collection: $COBOL_RAG_COLLECTION"
echo "Chroma dir: $COBOL_RAG_CHROMA_DIR"
echo

cobol-rag inspect "$TARGET_JSONL" --preview-chars 80
cobol-rag sync "$TARGET_JSONL" --apply

if [[ "$START_SERVER" -eq 1 ]]; then
  echo
  echo "Open: http://127.0.0.1:$PORT/"
  python -m uvicorn cobol_rag.api:app --host 127.0.0.1 --port "$PORT"
fi
