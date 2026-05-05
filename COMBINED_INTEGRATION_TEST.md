# COBOL-REKT Combination Test

This branch is for testing the combined PDCBVC evidence without replacing the stable RAG input.

## What This Contains

- `data/inbox/control_flow_rag_documents_combined.jsonl`
  - 824 existing control-flow RAG documents
  - 647 imported cobol-rekt `knowledge-base_rag` chunks
  - 5 generated integration summary/conflict documents
- `data/inbox/control_flow_rag_documents_combined.manifest.json`
  - provenance and chunk type counts for the combined JSONL

The current stable file remains untouched:

- `data/inbox/control_flow_rag_documents.jsonl`

## Analysis Side

Use the analysis branch that generated the combined artifacts:

```bash
cd legacy-program-analysis
git checkout feature/combine-cobol-rekt-analysis
git pull origin feature/combine-cobol-rekt-analysis
```

If the combined artifacts are not already present, regenerate them from the cobol-rekt bundle:

```bash
python scripts/pipeline/import_cobol_rekt_rag_bundle.py \
  --program PDCBVC \
  --cobol-rekt-rag-bundle /path/to/knowledge-base_rag \
  --final-scripts-root artifacts/final/final_scripts \
  --out-root artifacts/combined/final_scripts \
  --base-rag-jsonl ../cobol-rag-pipeline/data/inbox/control_flow_rag_documents.jsonl \
  --combined-rag-jsonl artifacts/combined/rag_index/control_flow_rag_documents_combined.jsonl
```

## RAG Side

```bash
cd cobol-rag-pipeline
git checkout feature/combine-cobol-rekt-rag
git pull origin feature/combine-cobol-rekt-rag
source .venv/bin/activate
pip install -e .
```

Point direct structured answers at the combined final scripts:

```bash
export COBOL_RAG_FINAL_SCRIPTS_DIR="../legacy-program-analysis/artifacts/combined/final_scripts"
```

Index only the combined JSONL test file:

```bash
cobol-rag inspect data/inbox/control_flow_rag_documents_combined.jsonl --preview-chars 80
cobol-rag sync data/inbox/control_flow_rag_documents_combined.jsonl --apply
```

Run Ollama and the UI:

```bash
ollama serve
ollama pull granite-code:8b-instruct
ollama pull mxbai-embed-large:latest
python -m uvicorn cobol_rag.api:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/
```

## Test Questions

```text
Which programs does PDCBVC call and with which parameters?
In PDCBVC.CBL, how does the program decide whether to execute BROWSE-FASE1 or BROWSE-FASE2?
In PDCBVC.CBL, under what exact conditions does the program read the semaphore PDAGGVIP?
In PDCBVC.CBL, explain the full sequence of operations during BROWSE-FASE1 before PDCBVC1 is sent.
In PDCBVC.CBL, how is pagination calculated and maintained when the user presses ENTER, PF7, or PF8?
In PDCBVC.CBL, when the user selects a row, how does the program validate the selected progressivo?
In PDCBVC.CBL, compare PF1, PF2, PF3, PF4, and PF9 behavior.
In PDCBVC.CBL, identify all paths that lead to an error message or abnormal termination.
Is there any unused code/copy in this PDCBVC?
How many unused copybooks?
```

## Expected Difference

The old JSONL is better for your existing direct PDCBVC facts. The combined JSONL adds your friend's deeper cobol-rekt chunks for control flow, paragraph logic, screen behavior, error paths, and call contracts.

If a fact conflicts, the combined analysis preserves both sources instead of overwriting one with the other.
