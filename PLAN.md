# COBOL RAG Pipeline Plan

## Goal

Build a flexible local RAG pipeline that can be used by both of us even when the input artifacts do not all have the same format.

The pipeline should make it easy to:

- add new analysis outputs by dropping files or folders into an inbox
- remove or refresh indexed files without deleting the whole database
- switch models, embedding models, vector collections, prompts, and retrieval settings from config
- start with CLI workflows and grow into a chat-style RAG interface
- keep answers grounded with citations back to the original artifact/chunk

Core stack:

- LlamaIndex for documents, indexing, retrieval, query engines, and chat engines
- ChromaDB as the local persistent vector database
- Ollama as the first local LLM and embedding provider
- Python CLI for ingestion, sync, removal, retrieval debug, query, and chat

## Design Principle

Separate the pipeline into small replaceable layers:

```text
input files -> loader adapter -> normalized documents -> index manager -> retriever -> answer/chat engine
```

This is important because my current COBOL artifacts and my friend's artifacts may not look the same. The stable contract should be the normalized `Document`, not one specific source format.

## Development Rule: Small, Safe, Documented Steps

Every implementation step should be small enough to verify with one command.

For each step:

1. Make the smallest useful code change.
2. Update `README.md` if the pipeline behavior, diagram, command, or extension point changed.
3. Update this plan if the build order or safety rule changed.
4. Run the narrowest useful verification command.
5. Move on only after the result is understandable.

This keeps the project easier to adjust when new artifact formats, models, prompts, or database workflows appear.

## Status Summary

Status labels:

- `Done`: implemented and verified
- `In Progress`: currently being built or partially implemented
- `Next`: the next small safe step
- `Planned`: not started yet

| Area | Status | Current verification |
| --- | --- | --- |
| Step 1: Local setup | Done | `.venv/bin/python --version` and dependencies installed |
| Step 2: Configuration | Done | `.venv/bin/cobol-rag config` prints active config |
| Step 3: LlamaIndex setup in code | Done | `.venv/bin/cobol-rag index-info` opens the configured Chroma collection |
| Step 4: Normalized document contract | Done | `cobol-rag inspect` shows normalized metadata without indexing |
| Step 5: Loader adapters | Done | general JSON and text loaders work; specific formats deferred |
| Step 6: Easy add/remove/update workflow | Done | `sync --dry-run`, `sync --apply`, then `sync --dry-run` proves add/update/skip |
| Step 7: Planned CLI | In Progress | `cobol-rag --help` and `cobol-rag config` work |
| Phase 1: Scaffold | Done | `.venv/bin/cobol-rag --help` and `.venv/bin/cobol-rag config` work |
| Phase 2: Loader system | Done | general loader registry, JSON loader, text loader, and inspect command work |
| Phase 3: Chroma index manager | In Progress | setup/open/info and document upsert work; list/delete helpers still pending |
| Phase 4: Manifest sync | Done | apply writes Chroma and manifest; follow-up dry-run reports skips |
| Phase 5: Remove/reset | Done | remove by source id/path and reset by collection work |
| Phase 6: Retrieval debug | Next | not implemented yet |
| Phase 7: One-shot query | Planned | not implemented yet |
| Phase 8: Chat RAG | Planned | not implemented yet |
| Phase 9: Evaluation harness | Planned | not implemented yet |

## Repository Layout

Create this structure:

```bash
pyproject.toml
README.md
.env.example
config/default.yaml
data/inbox/
data/archive/
data/manifests/
.chroma/
eval/questions.yaml
src/cobol_rag/
  __init__.py
  cli.py
  config.py
  loaders/
    __init__.py
    base.py
    generic_json.py
    plain_text.py
    registry.py
  index.py
  sync.py
  query.py
  chat.py
  evaluate.py
```

`data/inbox/` is the easy add/remove workflow. Put new output folders or files there, run one command, and the database updates.

The README must stay aligned with this architecture. It should include:

- a pipeline diagram
- a simple explanation of each pipeline step
- the current command for verifying each implemented slice
- clear future adjustment points for loaders, config, indexing, retrieval, and chat

## Step 1: Local Setup

Status: `Done`

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

If activation fails, the venv creation was probably interrupted. Re-run:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Pop OS/Ubuntu, do not run `python3 -m pip install ...` outside the venv. It will try to use the system Python and may fail with `externally-managed-environment`.

You can always bypass activation and use the venv directly:

```bash
.venv/bin/python -m pip install --upgrade pip
```

Install the first dependencies:

```bash
pip install \
  llama-index \
  llama-index-vector-stores-chroma \
  llama-index-llms-ollama \
  llama-index-embeddings-ollama \
  chromadb \
  typer \
  rich \
  pydantic-settings \
  pyyaml
```

Install Ollama separately, then pull initial local models:

```bash
ollama serve
ollama pull granite-code:8b
ollama pull mxbai-embed-large
```

The exact model names should stay configurable. Do not hard-code them into the pipeline.

## Step 2: Configuration

Status: `Done`

Use `config/default.yaml` for defaults:

```yaml
paths:
  chroma_dir: ".chroma"
  inbox_dir: "data/inbox"
  archive_dir: "data/archive"
  manifest_dir: "data/manifests"

llm:
  provider: "ollama"
  model: "granite-code:8b"
  base_url: "http://localhost:11434"
  request_timeout: 300
  temperature: 0.1

embedding:
  provider: "ollama"
  model: "mxbai-embed-large:latest"
  base_url: "http://localhost:11434"

index:
  collection: "cobol-dev"
  chunk_mode: "pre_chunked"
  batch_size: 64

retrieval:
  top_k: 6
  filters: {}
  similarity_cutoff: null

answers:
  require_citations: true
  show_sources: true
```

Allow overrides from CLI flags and environment variables:

```bash
cobol-rag query --config config/default.yaml --collection friend-test --top-k 10 "..."
COBOL_RAG_LLM_MODEL=llama3.1:8b cobol-rag chat
```

## Step 3: LlamaIndex Setup In Code

Status: `Done`

Use LlamaIndex integrations explicitly:

```python
import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

Settings.llm = Ollama(
    model=config.llm.model,
    base_url=config.llm.base_url,
    request_timeout=config.llm.request_timeout,
    temperature=config.llm.temperature,
)
Settings.embed_model = OllamaEmbedding(
    model_name=config.embedding.model,
    base_url=config.embedding.base_url,
)

db = chromadb.PersistentClient(path=config.paths.chroma_dir)
collection = db.get_or_create_collection(config.index.collection)
vector_store = ChromaVectorStore(chroma_collection=collection)
index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=Settings.embed_model,
)
```

For ingestion, create `Document` objects with stable `id_` values and metadata, then insert or refresh them through the index.

Implemented in:

```bash
src/cobol_rag/index.py
```

Verification:

```bash
cobol-rag index-info
```

## Step 4: Normalized Document Contract

Status: `Done`

Every loader must return LlamaIndex `Document` objects with:

```json
{
  "text": "...",
  "metadata": {
    "source_id": "stable-file-or-chunk-id",
    "source_path": "data/inbox/example.json",
    "source_format": "generic_json",
    "source_name": "example.json",
    "content_hash": "...",
    "metadata_version": "1"
  }
}
```

Rules:

- `source_id` must be stable across runs for the same logical chunk.
- `content_hash` decides whether a document needs re-indexing.
- `source_format` tells us which loader handled it.
- Specific fields such as `program`, `chunk_id`, `chunk_type`, `schema_version`, or tool-specific fields are optional and should be added by later format-specific loaders only when the source format actually provides them.

## Step 5: Loader Adapters

Status: `Done`

Implement loaders behind one interface:

```python
class LoaderAdapter:
    name: str

    def can_load(self, path: Path) -> bool:
        ...

    def load(self, path: Path) -> list[Document]:
        ...
```

Initial adapters:

- `generic_json`: reads general JSON objects/lists and extracts text from configurable fields when possible
- `plain_text`: reads `.md`, `.txt`, `.cbl`, `.cpy`, `.cob`, `.jcl`, and other simple text-like files

Deferred adapters:

- `cobol_rekt_chunks`: later, after the general loader contract is stable
- friend-specific formats: later, when an actual sample is available
- JCL/copybook/Cobol-specific enrichment: later, after the generic flow works end to end

The CLI should auto-detect the loader, but also allow forcing one:

```bash
cobol-rag inspect data/inbox/example.json
cobol-rag inspect data/inbox/example.txt --loader plain_text
```

## Step 6: Easy Add/Remove/Update Workflow

Status: `Done`

Preferred workflow:

```bash
cp -r /path/to/new/output data/inbox/
cobol-rag inspect data/inbox/
cobol-rag sync --dry-run
cobol-rag sync --apply
cobol-rag sync --dry-run
```

Implemented behavior:

- `sync --dry-run` scans `data/inbox/`
- uses only the general loader registry
- computes `content_hash` for each normalized document
- reads `data/manifests/<collection>.json` if it exists
- reports what would be added, updated, or skipped
- does not write to Chroma
- does not write the manifest
- `sync --apply` inserts new documents into Chroma
- `sync --apply` refreshes changed documents by deleting the old `source_id` records before inserting the fresh document
- leave unchanged documents alone
- write a local manifest to `data/manifests/<collection>.json` after successful indexing
- print a summary of added, updated, skipped, and failed items

Implemented removal options:

```bash
cobol-rag remove --source-path data/inbox/old-output.json
cobol-rag remove --source-id plain_text:data/inbox/example.txt
```

Deferred removal options:

```bash
cobol-rag remove --program PDB305.CBL
cobol-rag remove --source-format generic_json
```

For Chroma, deletion is currently done through stable `source_id` metadata. Later format-specific loaders should keep enough metadata in every node to delete by `program`, `source_format`, `chunk_type`, or other domain-specific fields.

Implemented reset options:

```bash
cobol-rag reset --dry-run
cobol-rag reset --apply
```

Reset affects only the configured collection and its manifest. It does not delete files from `data/inbox/`.

## Step 7: Planned CLI

Status: `In Progress`

```bash
cobol-rag --help
```

Inspection:

```bash
cobol-rag inspect data/inbox/example.json
cobol-rag inspect data/inbox/example.txt --loader plain_text
```

Sync:

```bash
cobol-rag sync --dry-run
cobol-rag sync --collection friend-test
cobol-rag sync --inbox data/inbox --dry-run
```

Direct ingest:

```bash
cobol-rag ingest /path/to/PDB305.CBL.report --collection test-pdb305
```

List:

```bash
cobol-rag list
cobol-rag list --collection friend-test
```

Remove:

```bash
cobol-rag remove --source-id plain_text:data/inbox/example.txt
cobol-rag remove --source-path data/inbox/friend-output.json
```

Retrieval debug:

```bash
cobol-rag retrieve --top-k 5 "Which CICS programs does PDB305 call?"
```

One-shot query:

```bash
cobol-rag query "Which CICS programs does PDB305 call?"
```

Interactive chat:

```bash
cobol-rag chat
cobol-rag chat --collection friend-test --top-k 8
```

Evaluation:

```bash
cobol-rag eval --questions eval/questions.yaml
```

## Step 8: Implementation Phases

### Phase 1: Scaffold

Status: `Done`

Create packaging, config, and CLI:

```bash
pyproject.toml
README.md
.env.example
config/default.yaml
src/cobol_rag/__init__.py
src/cobol_rag/cli.py
src/cobol_rag/config.py
```

Exit criterion:

```bash
cobol-rag --help
cobol-rag config
```

works and prints the active config.

Documentation criterion:

- `README.md` has the pipeline diagram.
- `README.md` explains what changed in this phase.
- `PLAN.md` still lists the next smallest safe step.

### Phase 2: Loader System

Status: `Done`

Implement:

```bash
src/cobol_rag/loaders/base.py
src/cobol_rag/loaders/generic_json.py
src/cobol_rag/loaders/plain_text.py
src/cobol_rag/loaders/registry.py
```

Exit criterion:

```bash
cobol-rag inspect data/inbox/example.json
cobol-rag inspect data/inbox/example.txt
```

prints document counts by loader, source format, and source path without indexing anything.

Documentation criterion:

- Add a README note for every supported input format.
- Document how to add a future loader without touching indexing/query code.

### Phase 3: Chroma Index Manager

Status: `In Progress`

Implement:

```bash
src/cobol_rag/index.py
```

Responsibilities:

- open a persistent Chroma collection: done
- configure LlamaIndex `Settings.llm` and `Settings.embed_model`: done
- insert documents with stable IDs
- delete documents by metadata or document ID
- expose `get_index()`, `upsert_documents()`, `delete_where()`, and `list_sources()`

Current partial verification:

```bash
cobol-rag index-info
```

Exit criterion:

```bash
cobol-rag ingest data/inbox/PDB305.CBL.report --collection test-pdb305
cobol-rag list --collection test-pdb305
```

shows `PDB305.CBL`.

Documentation criterion:

- Document where Chroma persists data.
- Document when a collection should be reset, especially after changing embedding models.

### Phase 4: Manifest-Based Sync

Status: `Done`

Implement:

```bash
src/cobol_rag/sync.py
```

Manifest example:

```json
{
  "collection": "cobol-dev",
  "sources": {
    "generic_json:data/inbox/example.json:0": {
      "source_id": "generic_json:data/inbox/example.json:0",
      "source_path": "data/inbox/example.json",
      "content_hash": "...",
      "source_format": "generic_json"
    }
  }
}
```

Exit criterion:

```bash
cobol-rag sync --dry-run
cobol-rag sync --apply
cobol-rag sync --dry-run
```

The first command previews add/update/skip counts and clearly says `indexing: no` and `manifest_write: no`.

The apply command writes Chroma and the manifest.

The final dry-run reports everything as skipped/unchanged.

Documentation criterion:

- Document the exact add/update workflow using `data/inbox/`.
- Document what `--dry-run` means before using destructive or refresh behavior.

### Phase 5: Remove And Reset

Status: `Done`

Implement:

- `remove --source-id`: done
- `remove --source-path`: done
- `remove --program`: deferred until format-specific metadata exists
- `remove --source-format`: planned
- `remove --chunk-id`: deferred until format-specific metadata exists
- `reset`: done for the configured collection

Exit criterion:

```bash
cobol-rag remove --source-path data/inbox/example.txt --dry-run
cobol-rag remove --source-path data/inbox/example.txt --apply
cobol-rag index-info
cobol-rag reset --dry-run
cobol-rag reset --apply
cobol-rag sync --dry-run
```

remove decreases document count and updates the manifest.

reset clears the configured collection and removes its manifest, while keeping `data/inbox/` untouched. After reset, `sync --dry-run` reports inbox files as `would_add`.

Documentation criterion:

- Document each removal mode and what metadata it depends on.
- Keep reset clearly labeled as collection-level destructive behavior.

### Phase 6: Retrieval Debug Mode

Status: `Planned`

Before trusting generated answers, add retrieval-only output:

```bash
cobol-rag retrieve \
  --collection test-pdb305 \
  --top-k 5 \
  "Which CICS programs does PDB305 call?"
```

Output should show:

- similarity score
- source ID
- source path
- program
- chunk type
- short text preview

Exit criterion:

The CICS query retrieves `PDB305.CBL:cics_operations` in the top results.

Documentation criterion:

- Document how to read retrieval debug output.
- Explain that retrieval is checked before generated answers are trusted.

### Phase 7: One-Shot Query With Citations

Status: `Planned`

Implement:

```bash
src/cobol_rag/query.py
```

Requirements:

- use LlamaIndex query engine
- inject a strict prompt requiring citations
- answer only from retrieved context
- show source IDs after the answer
- warn when no relevant source was retrieved

Exit criterion:

```bash
cobol-rag query --collection test-pdb305 "Which CICS programs does PDB305 call?"
```

returns an answer with sources.

Documentation criterion:

- Document the answer shape.
- Document the rule that answers without citations are not trusted.

### Phase 8: Chat RAG

Status: `Planned`

Implement:

```bash
src/cobol_rag/chat.py
```

First version:

- terminal chat loop
- collection selection
- configurable `top_k`
- conversation memory for follow-up questions
- citations on every answer
- `/sources` command to show last retrieved chunks
- `/reset` command to clear chat memory
- `/exit` command to quit

Example:

```bash
cobol-rag chat --collection cobol-dev
```

Documentation criterion:

- Document chat commands such as `/sources`, `/reset`, and `/exit`.
- Explain how chat memory differs from indexed source data.

Later version:

- small web UI
- source preview panel
- filters for program, artifact type, and friend/user dataset

### Phase 9: Evaluation Harness

Status: `Planned`

Create:

```bash
eval/questions.yaml
src/cobol_rag/evaluate.py
```

Example questions:

```yaml
- id: pdb305-purpose
  question: "What is the main purpose of PDB305?"
  expected_chunks:
    - "PDB305.CBL:program_summary"

- id: pdb305-cics-calls
  question: "Which CICS programs does PDB305 call?"
  expected_chunks:
    - "PDB305.CBL:cics_operations"

- id: pdb305-map
  question: "Which map does PDB305 use?"
  expected_chunks:
    - "PDB305.CBL:cics_operations"

- id: pdb305-analysis-risk
  question: "Why is analysis confidence only medium?"
  expected_chunks:
    - "PDB305.CBL:program_summary"
    - "PDB305.CBL:cobol_analysis_health"
```

Metrics:

- retrieval hit rate
- top-k hit rate
- answer has citation
- answer mentions low-confidence/degraded analysis when relevant
- answer refuses or says "not found" when sources do not support a claim

## First Test Data

Initial known report:

```bash
/home/eri/workspace/sapienza/master-thesis/cobol-rekt/out/report/PDB305.CBL.report
```

Suggested local test workflow:

```bash
mkdir -p data/inbox
cp -r /home/eri/workspace/sapienza/master-thesis/cobol-rekt/out/report/PDB305.CBL.report data/inbox/
cobol-rag sync --collection test-pdb305
cobol-rag retrieve --collection test-pdb305 "Which CICS programs does PDB305 call?"
cobol-rag query --collection test-pdb305 "Which CICS programs does PDB305 call?"
cobol-rag chat --collection test-pdb305
```

Good first questions:

```text
What is the main purpose of PDB305?
Which CICS programs does PDB305 LINK or XCTL to?
Which map does PDB305 use?
Which paragraphs handle PF7 and PF8?
Why is the analysis confidence only medium?
What copybooks were stubbed?
```

## Citation Rule

Every answer must include citations back to source IDs or chunk IDs.

Example answer shape:

```text
PDB305 links to PD0GCODA, PD0UTI01, PD1AC, PD3SORT, TE0CDUMP and XCTLs to PDPRED.

Sources:
- PDB305.CBL:cics_operations
```

If there is no citation, treat the answer as not trustworthy.

## Friend/Different Format Support

When my friend has a new format, do not rewrite the whole pipeline.

Add one loader adapter:

```bash
src/cobol_rag/loaders/friend_format.py
```

Then register it in the loader registry. The rest of the pipeline should continue to work because it receives normalized LlamaIndex `Document` objects.

If the format is mostly JSON, first try to support it through `generic_json` with config:

```yaml
loaders:
  generic_json:
    text_fields:
      - "text"
      - "content"
      - "summary"
    metadata_fields:
      - "program"
      - "file"
      - "section"
      - "kind"
```

## Later Extensions

After the selected-program flow works:

- add web chat UI
- add file watcher mode: `cobol-rag watch`
- add JCL chunks
- add copybook chunks
- add cross-program call graph chunks
- add hybrid retrieval with BM25 plus vectors
- add reranking
- add comparison between static-analysis answers and RAG answers
- add confidence-aware prompting using analysis health chunks
- export evaluation reports as Markdown/CSV

## Official LlamaIndex References Checked

- Installation: https://docs.llamaindex.ai/en/stable/getting_started/installation/
- Chroma vector store: https://docs.llamaindex.ai/en/stable/api_reference/storage/vector_store/chroma/
- Storing with vector stores: https://docs.llamaindex.ai/en/latest/understanding/storing/storing
- Ollama LLM integration: https://docs.llamaindex.ai/en/stable/api_reference/llms/ollama/
- Ollama embedding integration: https://docs.llamaindex.ai/en/stable/api_reference/embeddings/ollama/
- Document management: https://docs.llamaindex.ai/en/v0.10.23/module_guides/indexing/document_management/
