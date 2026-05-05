#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run RAG from the fixed analysis output.

This copies:
  <analysis-repo>/artifacts/final/final_scripts/output/rag_index/rag_documents.jsonl

to:
  data/inbox/control_flow_rag_documents.jsonl

Then it indexes that file and starts the UI.

Usage:
  ./scripts/run_fixed_input_rag.sh --program PDHASI06 --analysis-repo ../legacy-program-analysis

Options:
  --program NAME         Program to use for direct final_scripts answers.
  --analysis-repo PATH   Analysis repo path. Default: ../legacy-program-analysis, then ../control_flow.
  --port N               UI port. Default: 8000.
  --no-install           Do not create/use venv and pip install -e .
  --no-server            Only copy/inspect/sync; do not start uvicorn.
  --pull-models          Pull Ollama models before indexing.
  -h, --help             Show this help.
EOF
}

PROGRAM=""
ANALYSIS_REPO=""
PORT="8000"
INSTALL=1
START_SERVER=1
PULL_MODELS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --program)
      PROGRAM="${2:-}"
      shift 2
      ;;
    --analysis-repo)
      ANALYSIS_REPO="${2:-}"
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

if [[ -z "$PROGRAM" ]]; then
  echo "Missing --program." >&2
  usage
  exit 2
fi

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

SOURCE_JSONL="$ANALYSIS_REPO_ABS/artifacts/final/final_scripts/output/rag_index/rag_documents.jsonl"
FINAL_SCRIPTS="$ANALYSIS_REPO_ABS/artifacts/final/final_scripts/output/program_artifacts/programs/$PROGRAM_UPPER/artifacts"
TARGET_JSONL="data/inbox/control_flow_rag_documents.jsonl"

if [[ ! -f "$SOURCE_JSONL" ]]; then
  echo "Generated RAG JSONL not found: $SOURCE_JSONL" >&2
  echo "Run this in the analysis repo first:" >&2
  echo "  python scripts/pipeline/run_fixed_input.py --program $PROGRAM_UPPER --mode my" >&2
  exit 1
fi

if [[ ! -d "$FINAL_SCRIPTS" ]]; then
  echo "Final scripts artifact root not found: $FINAL_SCRIPTS" >&2
  exit 1
fi

mkdir -p data/inbox data/archive
if [[ -f "$TARGET_JSONL" ]]; then
  STAMP="$(date +%Y%m%d%H%M%S)"
  cp "$TARGET_JSONL" "data/archive/control_flow_rag_documents.$STAMP.jsonl"
fi
cp "$SOURCE_JSONL" "$TARGET_JSONL"

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

export COBOL_RAG_FINAL_SCRIPTS_DIR="$FINAL_SCRIPTS"
export COBOL_RAG_COLLECTION="cobol-fixed-${PROGRAM_LOWER}"
export COBOL_RAG_CHROMA_DIR="data/chroma-fixed-${PROGRAM_LOWER}"

echo
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
