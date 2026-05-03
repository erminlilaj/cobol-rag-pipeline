# PDCBVC RAG Evaluation Report

Date: 2026-05-03

Branch under test: `rag-factory-integration`

Base branch: `origin/cobol-rekt-wip`

Test program: `PDCBVC`

Input indexed for test:

```text
data/inbox/control_flow_rag_documents.jsonl
```

Generated from `control_flow` with compact chunks:

```text
rag_index_embed_safe_1200/rag_documents.jsonl
```

Local index state during test:

```text
collection: cobol-dev
indexed chunks: 666
embedding model: mxbai-embed-large:latest
```

The local Chroma DB, manifest, and copied inbox JSONL were test artifacts and are not intended to be committed.

## What Changed

1. Added a dedicated `rag_documents` loader.

   File:

   ```text
   src/cobol_rag/loaders/rag_documents.py
   ```

   Purpose:

   - Loads `rag_documents.jsonl` and `rag_documents.json` from the `control_flow` RAG factory.
   - Keeps the factory `text` as the document body.
   - Maps factory `type` into `chunk_type`.
   - Preserves useful metadata: `program`, `title`, `source_file`, `source_kind`, `chunk_index`, `chunk_count`, and factory ids.

2. Registered the loader before generic JSON.

   File:

   ```text
   src/cobol_rag/loaders/registry.py
   ```

   Reason:

   - Without this, factory JSONL was treated too generically and retrieval lost important COBOL artifact meaning.

3. Made `rag_index` folders resolve to `rag_documents.jsonl`.

   File:

   ```text
   src/cobol_rag/bundle.py
   ```

   Reason:

   - Passing a full `rag_index` folder should not recursively index `rag_documents.json`, manifests, and support files.
   - The authoritative ingestion file is `rag_documents.jsonl`.

4. Excluded metadata from embedding text.

   File:

   ```text
   src/cobol_rag/loaders/base.py
   ```

   Reason:

   - LlamaIndex was embedding metadata together with text.
   - Some chunks exceeded the Ollama embedding context length.
   - Metadata is still stored for filtering and citations, but not embedded.

5. Improved retrieval boosts for `control_flow` artifact types.

   File:

   ```text
   src/cobol_rag/retrieve.py
   ```

   Added first-class handling for types such as:

   ```text
   architecture.calls
   architecture.call
   architecture.copybooks
   architecture.db2_table
   architecture.sqlinclude
   global.call_target
   global.program_dependencies
   global.copybook_usage
   global.db2_table_usage
   program.comments
   program.comment
   program.summary
   ```

6. Updated README with the `control_flow` ingestion workflow and tested chunk sizing.

7. Added `*.egg-info/` to `.gitignore`.

## Evaluation Summary

| Question | Score | Result |
|---|---:|---|
| Which datasets and DB2 tables are used by PDCBVC? | 8/10 | Correctly found DB2 table `DUAL` and `selectIntoStatement`. Did not invent dataset output. SQL includes were useful but a little noisy. |
| Which variables connect to screen map field `M1MSGO`? | 8/10 | Correctly found `M1MSGO`, origin `COPY:PDCBVCM`, and modification paragraphs. Larger preview showed write-site evidence. |
| Which outside programs are used, and with which parameters? | 6.5/10 | Correctly found outside targets and call types. Parameter details were weak or absent in retrieval output. |
| Which program calls PDCBVC, and with which parameter? | 2/10 | Current index only contains one program, so incoming-call questions cannot be answered reliably. Returned outgoing calls instead. |
| Which dataset does PDCBVC produce? | 3/10 | No produced dataset evidence was found. Retrieval fell back to DB2 table evidence. |
| Is there unused code/copy in PDCBVC? | 6/10 | Commented-out code count was accurate. Unused copybook evidence was weak or missing. |
| Is there any forced value in PDCBVC, and for who? | 6.5/10 | A targeted query found some literal assignments, but mixed them with normal variable-to-variable moves and missed several important hardcoded values. |

Overall retrieval quality for this single-program test:

```text
6.5 to 7 / 10
```

Evidence retrieval is already useful for DB2/table usage, screen-map variable tracing, outgoing calls, and commented-out code. It is not yet strong enough for call parameters, produced datasets, incoming callers, unused copybooks, or complete forced-value analysis.

## Detailed Findings

### 1. DB2 Tables And Datasets

Question:

```text
Which datasets and DB2 tables are used by program PDCBVC?
```

Retrieved answer:

```text
DB2 table: DUAL
Statement type: selectIntoStatement
SQL includes: PDPSQLER, SQLCA, PDWSQLER
```

Assessment:

```text
Accurate for DB2 table usage.
No dataset usage was proven by the retrieved evidence.
```

Rating:

```text
8/10
```

### 2. Screen Field Data Trace

Question:

```text
In program PDCBVC, trace screen map field M1MSGO.
Which COBOL variables are connected to M1MSGO, and what is the data origin or computation?
```

Retrieved evidence:

```text
Variable: M1MSGO
Origin: COPY:PDCBVCM
Defined in: BROWSE-FASE1
Modified in:
- BROWSE-FASE1
- BROWSE-FASE2-NOSEL
- BROWSE-FASE2-NOTFND
- BROWSE-FASE2-TASTOER
- BROWSE-FASE2-VISUAL
Used in: none
Controls flow: no
```

Real source confirms writes such as:

```text
MOVE KFINE-DATI TO M1MSGO
MOVE WTASTOER TO M1MSGO
MOVE WMNOSEL TO M1MSGO
MOVE 'Progressivo digitato non presente' TO M1MSGO
```

Assessment:

```text
Good. It found the exact map field and origin. More targeted retrieval or an LLM answer is needed to summarize all write statements cleanly.
```

Rating:

```text
8/10
```

### 3. Outside Programs And Parameters

Question:

```text
Which outside programs and with which parameters are used in PDCBVC?
```

Retrieved outside targets:

```text
PD0UTI01  -> CICSLINKBYLITERAL
PD1FS00   -> CICSLINKBYLITERAL
PD1VOCI   -> CICSLINKBYLITERAL
PDPRED    -> CICSXCTLBYLITERAL
PXRSEMAF  -> CALLBYIDENTIFIER
```

Real source has parameter evidence that retrieval did not expose well:

```text
EXEC CICS LINK PROGRAM('PD1VOCI') COMMAREA(WPD1VOCI)
EXEC CICS LINK PROGRAM('PD1FS00') COMMAREA(WPD1FS00) LENGTH(PD1FS00-LUNGH)
CALL PXRSEMAF USING PXCSEMAF-AREA
EXEC CICS LINK PROGRAM('PD0UTI01') COMMAREA(WPDRUTI01)
EXEC CICS XCTL PROGRAM('PDPRED')
```

Assessment:

```text
Strong for target detection.
Weak for COMMAREA, LENGTH, and variable preparation.
```

Rating:

```text
Outside targets: 9/10
Parameters: 3/10
Overall: 6.5/10
```

### 4. Incoming Callers Of PDCBVC

Question:

```text
Which program calls PDCBVC and with which parameter is called?
```

Retrieved behavior:

```text
Returned outgoing calls from PDCBVC instead of incoming callers to PDCBVC.
```

Assessment:

```text
Expected limitation. The test index contains only PDCBVC, so incoming-call questions cannot be answered. This needs all programs indexed and a global incoming-call map.
```

Rating:

```text
2/10
```

### 5. Produced Datasets

Question:

```text
Which dataset does PDCBVC produce?
```

Retrieved behavior:

```text
Returned DB2 table DUAL and SQL include evidence.
No output dataset or report evidence was found.
```

Assessment:

```text
Weak for this question. PDCBVC appears to be a CICS program, not a batch dataset producer. The RAG should answer "no produced dataset found" instead of returning table evidence.
```

Rating:

```text
3/10
```

### 6. Unused Code Or Copy

Question:

```text
Is there any unused code or copy in PDCBVC?
```

Retrieved evidence:

```text
classification_counts.commented_out_code: 15
```

Real source confirms 15 commented-out code/data lines, including:

```text
* 03 LUNG PIC S9(4) COMP VALUE +32000.
*01 DFHBLLDS.
*** MOVE '20' TO PD1VOCI-TIPO-GEST.
*** ACCEPT WDATE FROM DATE.
**** MOVE SPACES TO TWCOB-XCTL-PGM
** MOVE PD1VOCI-TABVOX-AA34(PD1VOCI-IND)
** TO SESSIONE-VARIAZ-AA34.
** MOVE '/' TO SESSIONE-VARIAZ-B1.
** MOVE PD1VOCI-TABVOX-MESE(PD1VOCI-IND)
** TO SESSIONE-VARIAZ-MM.
*** EXEC CICS ADDRESS TWA(TWA-BLL) END-EXEC.
* PERFORM DELETE-TS THRU DELETE-TS-EXIT.
* MOVE 'P' TO TWCOB-FUNZIONE.
```

Copybook result:

```text
The RAG did not produce a reliable unused-copybook answer.
```

Assessment:

```text
Good for commented-out code.
Weak for unused copybooks.
```

Rating:

```text
6/10
```

### 7. Forced Values

Question:

```text
Is there any forced value in PDCBVC, and for who?
```

Initial result:

```text
Weak. It mostly returned CICS constants and normal variables.
```

Targeted query found some true literal assignments:

```text
MOVE SPACES TO FUNZ
MOVE '00' TO PD1VOCI-TIPO-GEST
MOVE 'A' TO PD1VOCI-TIPO-ESTRA
```

Real source contains more hardcoded assignments that retrieval did not fully surface:

```text
MOVE '03' TO PD1FS00-FUNZIONE
MOVE '00' TO PD1VOCI-TIPO-GEST
MOVE '11' TO PD1VOCI-FUNZIONE
MOVE '12' TO PD1VOCI-FUNZIONE
MOVE '02' TO PD1VOCI-FUNZIONE
MOVE 'A' TO PD1VOCI-TIPO-ESTRA
MOVE '1' TO PD1VOCI-TIPO-VOCE
MOVE '3' TO PD1VOCI-TIPO-VOCE
MOVE '2' TO PD1VOCI-TIPO-VOCE
MOVE '4' TO PD1VOCI-TIPO-VOCE
MOVE '0' TO PD1VOCI-TIPO-VOCE
MOVE 'GET' TO PXCSEMAF-REQ
MOVE 'PDAGGVIP' TO PXCSEMAF-NAME
MOVE 'SEMAFORO' TO PXCSEMAF-AGENT
MOVE 'PDCBVC' TO PXCSEMAF-CALLER
MOVE 'CICS' TO PXCSEMAF-CALLER-TYPE
MOVE '1' TO TWCOB-FASE
```

Assessment:

```text
Partial. The dataflow chunks contain some evidence, but forced values need a dedicated artifact.
```

Rating:

```text
6.5/10
```

## Main Gaps

1. Call parameter extraction is missing or not represented strongly enough.

   Needed artifact:

   ```text
   architecture.call_parameters.json
   ```

   Should include:

   ```text
   target program
   call type
   paragraph
   raw EXEC CICS or CALL statement
   PROGRAM(...)
   COMMAREA(...)
   LENGTH(...)
   RESP(...)
   USING variables
   variables moved into COMMAREA before call
   variables read after call
   ```

2. Forced-value retrieval needs a dedicated literal-assignment artifact.

   Needed artifact:

   ```text
   dataflow.literal_assignments.json
   ```

   Should include:

   ```text
   paragraph
   line
   literal value
   target variable
   statement
   whether target controls flow
   whether target is a screen/map field
   whether target is part of a call COMMAREA
   ```

3. Dataset/report questions need explicit JCL/file I/O artifacts.

   Needed artifacts:

   ```text
   architecture.file_io.json
   jcl.program_datasets.json
   ```

   Should distinguish:

   ```text
   input dataset
   output dataset
   report dataset
   temp dataset
   DB2 table
   SQL include
   CICS map
   ```

4. Incoming-call questions need a full-program global graph.

   This cannot be judged from a single-program index.

5. Unused copybook analysis needs an explicit result.

   Needed artifact:

   ```text
   architecture.unused_copybooks.json
   ```

   Should compare:

   ```text
   COPY statements
   variables referenced from each copybook
   procedural references
   CICS/map copybooks
   utility/error/state copybooks
   ```

## Recommended Next Step

Keep the RAG engine changes in this branch. Improve the `control_flow` factory next with these high-impact artifacts:

```text
architecture.call_parameters.json
dataflow.literal_assignments.json
architecture.file_io.json
jcl.program_datasets.json
architecture.unused_copybooks.json
```

After those are generated and indexed, rerun the same question set. The expected score should move from roughly `6.5-7/10` to `8.5+/10` for practical COBOL support questions.
