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

## 2. Generate The RAG JSONL From Scratch

Run one program:

```bash
cd legacy-program-analysis
git switch feature/combine-cobol-rekt-analysis
git pull

python scripts/pipeline/run_fixed_input.py --program PDHASI06 --mode my
```

Run every program folder:

```bash
python scripts/pipeline/run_fixed_input.py --mode my
```

The main generated file is:

```text
artifacts/final/final_scripts/output/rag_index/rag_documents.jsonl
```

The generated direct-answer artifacts are under:

```text
artifacts/final/final_scripts/output/program_artifacts/programs/<PROGRAM>/artifacts
```

## 3. Run RAG With The Generated JSONL

Run this in `cobol-rag-pipeline`:

```bash
cd cobol-rag-pipeline
git switch feature/combine-cobol-rekt-rag
git pull

./scripts/run_fixed_input_rag.sh --program PDHASI06 --analysis-repo ../legacy-program-analysis
```

This copies:

```text
../legacy-program-analysis/artifacts/final/final_scripts/output/rag_index/rag_documents.jsonl
```

to:

```text
data/inbox/control_flow_rag_documents.jsonl
```

Then it indexes the file and starts:

```text
http://127.0.0.1:8000/
```

## Combined Mode

If each program folder also has a `knowledge-base_rag/` bundle:

```bash
cd legacy-program-analysis
python scripts/pipeline/run_fixed_input.py --program PDHASI06 --mode both
```

Combined output is written under:

```text
artifacts/final/final_scripts/output/combined/
```

For now, use the existing combined test runner for that output, or copy the combined JSONL manually into the RAG repo for comparison.
