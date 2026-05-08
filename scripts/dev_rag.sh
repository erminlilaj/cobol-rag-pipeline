#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the development RAG workflow with one command.

Default:
  ./scripts/dev_rag.sh

Default behavior:
  - mode: combined
  - program: all fixed-input programs
  - analysis repo: ../legacy-program-analysis
  - build analysis output before indexing
  - index into cobol-combined-all / data/chroma-combined-all
  - start the UI on http://127.0.0.1:8000/

Focused one-program run:
  ./scripts/dev_rag.sh --program PDCBVC
  ./scripts/dev_rag.sh --program PDCBVC --mode combined --no-server

Options:
  --program NAME             Run one program. If omitted, run all fixed-input programs.
  --mode my|combined         RAG mode. Default: combined.
  --analysis-repo PATH       Analysis repo path. Default: ../legacy-program-analysis, then ../control_flow.
  --no-build-analysis        Do not run the analysis pipeline before indexing.
  --port N                   UI port. Default: 8000.
  --no-install               Do not create/use venv and pip install -e .
  --no-server                Only build/index; do not start uvicorn.
  --pull-models              Pull Ollama models before indexing.
  -h, --help                 Show this help.
EOF
}

PROGRAM=""
MODE="combined"
ANALYSIS_REPO=""
PORT="8000"
INSTALL=1
START_SERVER=1
PULL_MODELS=0
BUILD_ANALYSIS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --program)
      PROGRAM="${2:-}"
      shift 2
      ;;
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --analysis-repo)
      ANALYSIS_REPO="${2:-}"
      shift 2
      ;;
    --no-build-analysis)
      BUILD_ANALYSIS=0
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

PYTHON_BIN="python3"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

resolve_path() {
  local path="$1"
  "$PYTHON_BIN" - "$path" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
}

if [[ -z "$ANALYSIS_REPO" ]]; then
  if [[ -d "../legacy-program-analysis" ]]; then
    ANALYSIS_REPO="../legacy-program-analysis"
  elif [[ -d "../control_flow" ]]; then
    ANALYSIS_REPO="../control_flow"
  else
    ANALYSIS_REPO="../legacy-program-analysis"
  fi
fi

if [[ -n "$PROGRAM" ]]; then
  args=(
    "$SCRIPT_DIR/run_fixed_input_rag.sh"
    --program "$PROGRAM"
    --mode "$MODE"
    --analysis-repo "$ANALYSIS_REPO"
    --port "$PORT"
  )
  if [[ "$BUILD_ANALYSIS" -eq 1 ]]; then
    args+=(--build-analysis)
  fi
  if [[ "$INSTALL" -eq 0 ]]; then
    args+=(--no-install)
  fi
  if [[ "$START_SERVER" -eq 0 ]]; then
    args+=(--no-server)
  fi
  if [[ "$PULL_MODELS" -eq 1 ]]; then
    args+=(--pull-models)
  fi
  exec "${args[@]}"
fi

ANALYSIS_REPO_ABS="$(resolve_path "$ANALYSIS_REPO")"

if [[ "$BUILD_ANALYSIS" -eq 1 ]]; then
  ANALYSIS_MODE="my"
  if [[ "$MODE" == "combined" ]]; then
    ANALYSIS_MODE="both"
  fi
  echo
  echo "Building analysis output for all programs: $ANALYSIS_MODE"
  (
    cd "$ANALYSIS_REPO_ABS"
    "$PYTHON_BIN" scripts/pipeline/run_fixed_input.py --mode "$ANALYSIS_MODE"
  )
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
  ollama pull granite4.1:8b
  ollama pull mxbai-embed-large:latest
fi

COBOL_RAG_BIN="cobol-rag"
RUN_PYTHON="python"
if [[ -x ".venv/bin/cobol-rag" ]]; then
  COBOL_RAG_BIN=".venv/bin/cobol-rag"
fi
if [[ -x ".venv/bin/python" ]]; then
  RUN_PYTHON=".venv/bin/python"
fi

export COBOL_RAG_COLLECTION="cobol-${MODE}-all"
export COBOL_RAG_CHROMA_DIR="data/chroma-${MODE}-all"
export COBOL_RAG_LLM_POLISH_FINAL_SCRIPTS="${COBOL_RAG_LLM_POLISH_FINAL_SCRIPTS:-false}"

declare -a INPUTS=()
if [[ "$MODE" == "combined" ]]; then
  INPUT_DIR="$ANALYSIS_REPO_ABS/artifacts/final/final_scripts/output/combined/rag_index"
  FINAL_SCRIPTS_ROOT="$ANALYSIS_REPO_ABS/artifacts/final/final_scripts/output/combined/final_scripts"
  shopt -s nullglob
  INPUTS=("$INPUT_DIR"/*_combined.jsonl)
  shopt -u nullglob
else
  INPUTS=("$ANALYSIS_REPO_ABS/artifacts/final/final_scripts/output/rag_index/rag_documents.jsonl")
  FINAL_SCRIPTS_ROOT="$ANALYSIS_REPO_ABS/artifacts/final/final_scripts/output/program_artifacts/programs"
fi

if [[ "${#INPUTS[@]}" -eq 0 ]]; then
  echo "No RAG JSONL files found for mode '$MODE'." >&2
  exit 1
fi

if [[ ! -d "$FINAL_SCRIPTS_ROOT" ]]; then
  echo "Final scripts root does not exist: $FINAL_SCRIPTS_ROOT" >&2
  exit 1
fi

export COBOL_RAG_FINAL_SCRIPTS_DIR="$FINAL_SCRIPTS_ROOT"

echo
echo "Mode: $MODE"
echo "Program: all"
echo "Analysis repo: $ANALYSIS_REPO_ABS"
echo "Final scripts: $COBOL_RAG_FINAL_SCRIPTS_DIR"
echo "Collection: $COBOL_RAG_COLLECTION"
echo "Chroma dir: $COBOL_RAG_CHROMA_DIR"
echo "Inputs: ${#INPUTS[@]}"
echo

for input in "${INPUTS[@]}"; do
  if [[ ! -f "$input" ]]; then
    echo "Input does not exist: $input" >&2
    exit 1
  fi
  "$COBOL_RAG_BIN" inspect "$input" --preview-chars 80
  "$COBOL_RAG_BIN" sync "$input" --apply
done

if [[ "$START_SERVER" -eq 1 ]]; then
  echo
  echo "Open: http://127.0.0.1:$PORT/"
  "$RUN_PYTHON" -m uvicorn cobol_rag.api:app --host 127.0.0.1 --port "$PORT"
fi
