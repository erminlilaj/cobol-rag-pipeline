# Evaluation

This folder contains deterministic evaluation tools for the COBOL RAG assistant.

## Gold Suite

The main gold file is:

```bash
eval/questions/pdcbvc_gold.json
```

It defines question cases, categories, runtime requirements, and assertions:

- `contains_all`: every listed fragment must appear in the answer.
- `contains_any`: at least one fragment from each group must appear.
- `regex_all`: every regex must match the answer.
- `forbidden`: listed fragments must not appear.
- `min_sources`: minimum number of retrieved sources required.
- `source_contains_all`: fragments that must appear in retrieved source text or metadata.

Cases can require:

- `final_scripts`: structured analysis artifacts are discoverable.
- `rag_index`: a local manifest or Chroma directory exists.
- `ollama`: the configured Ollama server responds.

Cases whose requirements are missing are skipped by default, so the structured
final_scripts tests can run offline while RAG/LLM tests are still tracked.

## Run

From the repository root:

```bash
python eval/run_gold_eval.py
```

Useful options:

```bash
python eval/run_gold_eval.py --category control_flow
python eval/run_gold_eval.py --case calls.all_parameters --show-answers
python eval/run_gold_eval.py --json-output eval/out/pdcbvc_gold_report.json
python eval/run_gold_eval.py --markdown-output eval/out/pdcbvc_gold_report.md
```

For this project, point the runner at the detailed artifacts if they are not
inside the repository:

```bash
export COBOL_RAG_FINAL_SCRIPTS_DIR="$PWD/final_scripts"
python eval/run_gold_eval.py
```

On Windows PowerShell:

```powershell
$env:COBOL_RAG_FINAL_SCRIPTS_DIR="C:\path\to\final_scripts"
python eval\run_gold_eval.py
```

## Artifact Enrichment

Before evaluating review-style questions, preview the normalized artifacts that
the assistant can derive from `final_scripts`:

```bash
cobol-rag enrich-final-scripts --root /path/to/final_scripts --program PDCBVC --dry-run
```

The current enrichment covers:

- `quality.dead_code` for commented-out code and CFG reachability.
- `architecture.unused_copybooks` for COPY members with no current reference evidence.
- `jcl.file_io` for JCL DD read/write/SYSOUT evidence when a batch program is linked.
- `screen_field_lineage` for BMS/map fields, field families, connected variables,
  and exact read/write/control evidence.

Use `--apply` only when you want those generated JSON files written into the
bundle. The direct-answer layer can still build them in memory for local tests.

Gold cases should assert both facts and citations. For example, a screen-field
answer should mention the field, the map copybook, connected variables, and a
source such as `dataflow.variable/dataflow.variable.SCELTAI.json | line 317`.
