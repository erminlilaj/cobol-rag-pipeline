#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run one RAG test mode with minimal setup.

Usage:
  ./scripts/run_rag_test.sh --mode my --program NEWPROG --analysis-repo ../legacy-program-analysis
  ./scripts/run_rag_test.sh --mode combined --program NEWPROG --analysis-repo ../legacy-program-analysis
  ./scripts/run_rag_test.sh --mode rekt --program NEWPROG --rekt-bundle /path/to/knowledge-base_rag

Options:
  --mode my|rekt|combined     Required test mode.
  --program NAME              Required COBOL program name.
  --analysis-repo PATH        Analysis repo path. Default: ../legacy-program-analysis, then ../control_flow.
  --rekt-bundle PATH          cobol-rekt knowledge-base_rag bundle. Required for --mode rekt.
  --port N                    UI port. Default: 8000.
  --no-install                Do not create/use venv and pip install -e .
  --no-server                 Only inspect/sync; do not start uvicorn.
  --pull-models               Pull Ollama LLM and embedding models before indexing.
  -h, --help                  Show this help.
EOF
}

MODE=""
PROGRAM=""
ANALYSIS_REPO=""
REKT_BUNDLE=""
PORT="8000"
INSTALL=1
START_SERVER=1
PULL_MODELS=0

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
    --rekt-bundle)
      REKT_BUNDLE="${2:-}"
      shift 2
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

if [[ -z "$MODE" || -z "$PROGRAM" ]]; then
  echo "Missing --mode or --program." >&2
  usage
  exit 2
fi

case "$MODE" in
  my|rekt|combined) ;;
  *)
    echo "--mode must be one of: my, rekt, combined" >&2
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

resolve_dir() {
  local path="$1"
  python - "$path" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
}

PROGRAM_UPPER="$(printf '%s' "$PROGRAM" | tr '[:lower:]' '[:upper:]')"
PROGRAM_LOWER="$(printf '%s' "$PROGRAM" | tr '[:upper:]' '[:lower:]')"
ANALYSIS_REPO_ABS="$(resolve_dir "$ANALYSIS_REPO")"

INPUT=""
FINAL_SCRIPTS=""

case "$MODE" in
  my)
    INPUT="$ANALYSIS_REPO_ABS/artifacts/experiments/$PROGRAM_UPPER/my_analysis/rag_index/rag_documents.jsonl"
    FINAL_SCRIPTS="$ANALYSIS_REPO_ABS/artifacts/experiments/$PROGRAM_UPPER/my_analysis/program_artifacts/programs/$PROGRAM_UPPER/artifacts"
    ;;
  combined)
    INPUT="$ANALYSIS_REPO_ABS/artifacts/experiments/$PROGRAM_UPPER/combined/rag_index/${PROGRAM_UPPER}_combined.jsonl"
    FINAL_SCRIPTS="$ANALYSIS_REPO_ABS/artifacts/experiments/$PROGRAM_UPPER/combined/final_scripts"
    ;;
  rekt)
    if [[ -z "$REKT_BUNDLE" ]]; then
      echo "--mode rekt requires --rekt-bundle /path/to/knowledge-base_rag" >&2
      exit 2
    fi
    INPUT="$(resolve_dir "$REKT_BUNDLE")"
    FINAL_SCRIPTS=""
    ;;
esac

if [[ ! -e "$INPUT" ]]; then
  echo "Input does not exist: $INPUT" >&2
  exit 1
fi

if [[ -n "$FINAL_SCRIPTS" && ! -d "$FINAL_SCRIPTS" ]]; then
  echo "Final scripts root does not exist: $FINAL_SCRIPTS" >&2
  exit 1
fi

PYTHON_BIN="python3"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

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

export COBOL_RAG_COLLECTION="cobol-${MODE}-${PROGRAM_LOWER}"
export COBOL_RAG_CHROMA_DIR="data/chroma-${MODE}-${PROGRAM_LOWER}"

if [[ -n "$FINAL_SCRIPTS" ]]; then
  export COBOL_RAG_FINAL_SCRIPTS_DIR="$FINAL_SCRIPTS"
else
  unset COBOL_RAG_FINAL_SCRIPTS_DIR || true
fi

echo
echo "Mode: $MODE"
echo "Program: $PROGRAM_UPPER"
echo "Input: $INPUT"
echo "Collection: $COBOL_RAG_COLLECTION"
echo "Chroma dir: $COBOL_RAG_CHROMA_DIR"
if [[ -n "${COBOL_RAG_FINAL_SCRIPTS_DIR:-}" ]]; then
  echo "Final scripts: $COBOL_RAG_FINAL_SCRIPTS_DIR"
fi
echo

cobol-rag inspect "$INPUT" --preview-chars 80
cobol-rag sync "$INPUT" --apply

if [[ "$START_SERVER" -eq 1 ]]; then
  echo
  echo "Open: http://127.0.0.1:$PORT/"
  python -m uvicorn cobol_rag.api:app --host 127.0.0.1 --port "$PORT"
fi
