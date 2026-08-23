# Current COBOL RAG System Assessment

Date: 2026-05-04

Scope:

- UI branch worktree: `.worktrees/ui`
- Main RAG package: `src/cobol_rag`
- Current inbox index file: `data/inbox/control_flow_rag_documents.jsonl`
- Current test program: `PDCBVC`

## Immediate Finding

The question:

```text
how many code lines is PDCBVC
```

failed because it was treated as a normal RAG + LLM question. The raw analysis artifact already had the answer:

```text
program.comments.json -> metrics.total_lines: 912
```

The UI now has a deterministic metadata answer path. It answers:

```text
PDCBVC has 912 total source lines. The comments artifact also reports 65 comment lines.
15 of those are classified as commented-out code. Source: `program.comments.json` metrics.
```

This kind of basic factual question should not depend on the LLM.

## Current Index Shape

Current `control_flow_rag_documents.jsonl` contains:

```text
671 total chunks
577 PDCBVC chunks
94 global chunks
```

Largest indexed artifact groups:

```text
dataflow.variable: 257
dataflow.used_variables: 96
program.comments: 93
program.comment: 40
business_rule: 36
global.control_variable_usage: 33
global.shared_variables.summary: 24
controlflow.cfg: 23
```

Important artifacts present:

```text
architecture.call_parameters
architecture.calls
architecture.copybooks
architecture.db2_table
architecture.sqlinclude
controlflow.cfg
dataflow.literal_assignments
dataflow.used_variables
program.comments
program.summary
ui.cics.navigation
global copybook/call/db2/shared-variable maps
```

Important artifacts missing:

```text
unused_copybooks
architecture.file_io
jcl.program_datasets
global incoming-call graph across multiple programs
dedicated program.metrics summary
```

## Current Quality Rating

Overall practical quality after adding the `final_scripts` direct-answer layer:

```text
7.5 / 10 for PDCBVC single-program questions
```

Breakdown:

```text
Artifact coverage for one CICS program: 7.5 / 10
Retrieval quality when Ollama embedding is running: 6.5-7 / 10
Answer reliability for basic final_scripts facts: 8 / 10
Runtime reliability: 5 / 10
UI usability: 6.5 / 10
Multi-program system understanding: 3 / 10
```

The system is useful, but not yet robust. It has enough analysis artifacts to answer many COBOL questions, but it still relies too much on retrieval + LLM for facts that should be deterministic.

## What Works Well

Good current capabilities:

```text
Outgoing calls from PDCBVC
Call parameters after adding architecture.call_parameters
Copybooks used by PDCBVC
DB2 table evidence
Control-flow edges and paragraph transitions
Screen/map field variable tracing
Used-variable and shared-variable evidence
Literal/forced assignments after adding dataflow.literal_assignments
Commented-out code from program.comments
Basic line count after the deterministic metadata fix
Direct final_scripts answers before vector retrieval
```

Examples of questions that should now be answerable:

```text
how many code lines is PDCBVC
Is there commented-out code in PDCBVC?
Which programs does PDCBVC call?
Which parameters are used when PDCBVC calls outside programs?
Which variables feed M1MSGO?
Which DB2 tables does PDCBVC use?
Which forced values are assigned in PDCBVC?
Which COPY members are used by PDCBVC?
Which business rules are in PDCBVC?
Which PF keys does PDCBVC handle?
```

## Final Scripts Direct Answers Added

The chat now checks `final_scripts` JSON before vector retrieval for high-confidence facts.

Covered direct-answer categories:

```text
program line/LOC/comment counts
outgoing calls and parameters
commented-out code/data
literal/forced assignments
copybook list and categories
copybook usage heuristic
DB2 tables and SQL includes
JCL/dataset availability for the program
CICS UI/PF-key navigation
business rule list
```

Smoke-tested examples:

```text
how many code lines is PDCBVC
Which programs does PDCBVC call and with which parameters?
Is there any unused code/copy in this PDCBVC?
Is there any forced value in PDCBVC?
Which dataset does PDCBVC produce?
Which copybooks are used by PDCBVC?
Which DB2 tables does PDCBVC use?
what business rules are in PDCBVC
what PF keys does PDCBVC handle?
```

These now answer without requiring Ollama LLM generation.

## What Is Missing

### 1. Runtime Health

If Ollama is not running, RAG retrieval can fail because the query embedding model is also served by Ollama:

```text
mxbai-embed-large:latest
```

The UI now returns a cleaner message for this, but the system still needs:

```text
Ollama health indicator
embedding model health indicator
LLM model health indicator
clear "RAG unavailable" vs "LLM unavailable" states
```

### 2. Deterministic Fact Answers

Questions like these should not go through the LLM:

```text
how many lines?
how many comments?
how many copybooks?
how many calls?
what is the program name?
what artifacts are indexed?
```

We added line count, but more metadata answerers are needed.

### 3. Unused Copybooks

The system can say which copybooks are used or missing from packages, but it cannot prove unused copybooks yet.

Current improvement:

```text
The direct-answer layer now gives a heuristic by comparing listed COPY members against dataflow origins,
literal assignment prefixes, and call parameter prefixes.
```

For `PDCBVC`, the current heuristic reports:

```text
COPY members with referenced variables:
DFHAID, PD1FS00, PD1VOCI, PDCBVCM, PDIABEND, PDRTWA2, PDRUTI01, PDSAVTW2, PXCSEMAF

Need review / possibly unused by this heuristic:
DFHBMSCA, PDRTIP01, PDRVC
```

This is better than no answer, but it is still not a proof.

Needed artifact:

```text
architecture.unused_copybooks.json
```

### 4. Produced Datasets / File I/O

The system does not have a strong answer for:

```text
Which dataset does PDCBVC produce?
```

Needed artifacts:

```text
architecture.file_io.json
jcl.program_datasets.json
```

Current finding:

```text
final_scripts contains detailed JCL artifacts for jobs such as PDCAFIN2, PDADDRE1, and PDASCO01.
No JCL summary currently connects PDCBVC to produced datasets.
For PDCBVC, the direct answer now says no produced dataset evidence was found.
```

### 5. Incoming Callers

The current index mostly contains one program. Therefore incoming-call questions cannot be answered reliably.

Needed:

```text
all-program index
global incoming-call graph
caller -> callee -> parameter map
```

### 6. Automated Evaluation

The project has a manual evaluation report, but it needs a repeatable golden-question test suite.

Needed:

```text
eval question YAML/JSON
expected evidence checks
expected answer fragments
CI/local command to run the suite
score report generated automatically
```

## Priority Fix Plan

P0:

```text
Add UI health checks for Ollama embedding and LLM models.
Expand deterministic answerers for all high-confidence final_scripts facts.
Add an automated golden-question eval suite.
```

P1:

```text
Add program.metrics artifact in the factory.
Add architecture.unused_copybooks artifact.
Add architecture.file_io and jcl.program_datasets artifacts.
Improve source snippets/citations for direct answers.
Push final_scripts direct-answer summaries into rag_documents.jsonl as first-class chunks.
```

P2:

```text
Index multiple programs.
Generate global incoming-call map.
Improve UI source display for long evidence.
```

## Bottom Line

The system is not bad, but it is not production-reliable yet.

It is strongest when a question maps directly to a strong artifact:

```text
calls
copybooks
variables
control flow
comments
literal assignments
DB2 table evidence
```

It is weakest when the needed artifact does not exist, or when a basic metadata fact is left to the LLM.

The next major improvement is not a bigger model. It is better deterministic routing plus missing analysis artifacts.
