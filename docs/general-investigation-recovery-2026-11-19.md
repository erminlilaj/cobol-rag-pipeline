# General investigation recovery — 19 November 2026

## Evidence behind the change

The 19-Nov exported chat showed two valid questions about item computations.
One investigation retrieved conditional source assignments but later failed on
tool protocol and citation-list mismatches. The other repeatedly read outlines
without opening the source body and returned an over-broad unavailable answer.
Neither trace established that PDHVAL51 had no relevant implementation.

## General repairs

- Normalize an empty optional relationship list to omission. Required fields,
  filters, malformed roles and substantive request restrictions remain strict.
- Repair scalar outline terms, including explicit comma-separated identifier
  lists. Preserve free-text and quoted/punctuation-bearing search phrases.
- Describe tool argument types and enum values in the model-facing help. Invalid
  numeric description depth remains an error, with the existing bounded isolated
  repair path; the executor does not guess what a numeric depth meant.
- Reconcile registered inline evidence IDs for singleton, grouped and range
  citations. Unknown IDs and malformed/oversized ranges still fail validation.
- Measure progress using evidence facts rather than pagination or handle churn.
- If a narrative investigation repeats navigation without progress, open at
  most two unread outline hit contexts per recovery round, each at most 120
  physical lines. Existing tool-count and wall-clock limits still apply. No
  corpus-wide scan or extra model budget is introduced.
- Explicitly label outline results as navigation, not absence or behavior proof.
  Detailed behavior answers citing only outlines must inspect actual evidence.
  Paragraph inventories and counts are not forced into body reading.
- Retain bounded registered source evidence across observation rotation and
  partial synthesis. Round-robin program selection prevents one owner from
  crowding out the other. Truncation is explicit; missing lines remain unknown.
- Explain fixed MOVE assignments, initialization, lookup and call-based behavior
  as well as arithmetic formulas. Preserve useful supported program findings
  when another program is unresolved; do not interpret tool failures or zero
  outline matches as source absence.

The independent factual reviewer, citation checks and coverage requirements
remain enabled. Memory, model selection, temperature, analysis, index format and
input data are unchanged.

## Verification

24 new offline regressions cover unrelated business identifiers and question
types, real protocol validation plus read-only execution and answer review,
typed repair, source-body recovery, evidence retention, bounded reads, pagination,
and rejection of unknown citations and contradicted partial answers.

The pre-existing investigation suites (214 tests) pass too. The wider baseline
has 21 failing legacy planner/router tests; compare the unchanged HEAD rather
than treating those as new regressions. A live Qwen run is still required:
mocked model decisions prove executor/recovery behavior, not model quality.

## Live test without reanalysis

Stop any active question before replacing the API. Pull the RAG test branch and
run the platform's existing start helper. Do not rerun setup, analysis or sync
for this RAG-only change. Keep the same model and memory disabled.

Test the original Italian and English questions, a single-program fixed-value
question, a dynamic-call question and a JCL job/step question. Export Download +
Debug and check actual body reads, supported per-program findings, explicit gaps,
citations, model/tool calls, tokens, time, and no repeated navigation loop. A
fluent accepted answer is not by itself proof of factual correctness.
