# Fixed Input Workflow

Use this when you want the whole test to be reproducible from GitHub.

## 1. Put Files In The Analysis Repo

In `legacy-program-analysis`, put every program under:

```text
artifacts/final/final_scripts/input/
  PDHASI06/
    PDHASI06.CBL
    PDHASI06_result.csv
    PDHASI06_controlflow.json
    copybooks/
      COPY1.cpy
      COPY2.cpy
    jcl/
      optional jobs
    knowledge-base_rag/
      optional cobol-rekt bundle for combined mode
```

Required for Hamza analysis:

- COBOL source file
- MAPA result `.csv` or `.txt`
- controlflow `.json`
- `copybooks/`

## 2. Generate And Run RAG With One Command

Run this in `cobol-rag-pipeline` for the default local development workflow:

```bash
./scripts/dev_rag.sh
```

This builds all fixed-input programs from `../legacy-program-analysis` in
combined mode, indexes every generated combined JSONL into
`cobol-combined-all`, and starts the UI at:

```text
http://127.0.0.1:8000/
```

For one program:

```bash
./scripts/dev_rag.sh --program PDHASI06 --mode combined --analysis-repo ../legacy-program-analysis
```

For indexing only:

```bash
./scripts/dev_rag.sh --no-server
```

## 3. Generate The Analysis Output Manually

Run one program:

```bash
cd legacy-program-analysis
git switch feature/combine-cobol-rekt-analysis
git pull

python scripts/pipeline/run_fixed_input.py --program PDHASI06 --mode both
```

Run every program folder:

```bash
python scripts/pipeline/run_fixed_input.py --mode both
```

The main generated file is:

```text
artifacts/final/final_scripts/output/rag_index/rag_documents.jsonl
```

Combined RAG files are generated under:

```text
artifacts/final/final_scripts/output/combined/rag_index/<PROGRAM>_combined.jsonl
```

The generated direct-answer artifacts are under:

```text
artifacts/final/final_scripts/output/program_artifacts/programs/<PROGRAM>/artifacts
```

## 4. Run One Program With The Lower-Level Script

Run this in `cobol-rag-pipeline`:

```bash
cd cobol-rag-pipeline
git switch feature/combine-cobol-rekt-rag
git pull

./scripts/run_fixed_input_rag.sh --mode combined --program PDHASI06 --analysis-repo ../legacy-program-analysis --build-analysis
```

In combined mode this copies:

```text
../legacy-program-analysis/artifacts/final/final_scripts/output/combined/rag_index/PDHASI06_combined.jsonl
```

to:

```text
data/inbox/control_flow_rag_documents_combined.jsonl
```

Then it indexes the file and starts:

```text
http://127.0.0.1:8000/
```

## Combined Mode Inputs

If each program folder also has a `knowledge-base_rag/` bundle:

```bash
cd legacy-program-analysis
python scripts/pipeline/run_fixed_input.py --program PDHASI06 --mode both
```

Combined output is written under:

```text
artifacts/final/final_scripts/output/combined/
```

`./scripts/dev_rag.sh` indexes that output directly. Manual copying is no longer
needed for normal combined-mode testing.
