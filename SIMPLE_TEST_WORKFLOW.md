# Simple Test Workflow

This is the short workflow for testing a new COBOL program in three modes:

- `my`: Hamza analysis only
- `rekt`: cobol-rekt bundle only
- `combined`: Hamza analysis plus cobol-rekt evidence

## 1. Prepare The Test Folder

Use one folder per program:

```text
new_test/
  cbl/
    NEWPROG.CBL
  copybooks/
    COPY1.cpy
    COPY2.cpy
  mapa/
    NEWPROG_result.csv
  controlflow/
    NEWPROG.json
  jcl/
    optional jobs
  knowledge-base_rag/
    optional cobol-rekt bundle
```

The Hamza pipeline needs `cbl`, `copybooks`, `mapa`, and `controlflow`.
The `jcl` folder is optional.

The cobol-rekt-only mode needs a `knowledge-base_rag` bundle.

## 2. Generate Hamza And Combined Artifacts

Run this from the analysis repo:

```bash
cd legacy-program-analysis
git switch feature/combine-cobol-rekt-analysis
git pull

python scripts/pipeline/run_program_test_case.py \
  --program NEWPROG \
  --case-root /path/to/new_test \
  --cobol-rekt-rag-bundle /path/to/new_test/knowledge-base_rag \
  --mode both \
  --recursive
```

This writes outputs under:

```text
artifacts/experiments/NEWPROG/
```

Important outputs:

```text
artifacts/experiments/NEWPROG/my_analysis/rag_index/rag_documents.jsonl
artifacts/experiments/NEWPROG/my_analysis/program_artifacts/programs/NEWPROG/artifacts
artifacts/experiments/NEWPROG/combined/rag_index/NEWPROG_combined.jsonl
artifacts/experiments/NEWPROG/combined/final_scripts
```

## 3. Run The RAG UI

Run this from the RAG repo:

```bash
cd cobol-rag-pipeline
git switch feature/combine-cobol-rekt-rag
git pull
```

Test Hamza analysis only:

```bash
./scripts/run_rag_test.sh --mode my --program NEWPROG --analysis-repo ../legacy-program-analysis
```

Test cobol-rekt only:

```bash
./scripts/run_rag_test.sh --mode rekt --program NEWPROG --rekt-bundle /path/to/new_test/knowledge-base_rag
```

Test combined:

```bash
./scripts/run_rag_test.sh --mode combined --program NEWPROG --analysis-repo ../legacy-program-analysis
```

The script creates/uses `.venv`, installs the RAG app, sets isolated Chroma
collections, indexes the chosen input, and starts the UI.

Open:

```text
http://127.0.0.1:8000/
```

## Useful Options

Index without starting the UI:

```bash
./scripts/run_rag_test.sh --mode combined --program NEWPROG --analysis-repo ../legacy-program-analysis --no-server
```

Pull Ollama models before running:

```bash
./scripts/run_rag_test.sh --mode combined --program NEWPROG --analysis-repo ../legacy-program-analysis --pull-models
```

Use another UI port:

```bash
./scripts/run_rag_test.sh --mode combined --program NEWPROG --analysis-repo ../legacy-program-analysis --port 8001
```
