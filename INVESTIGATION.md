# Model-led investigation

This optional workflow lets the model choose and combine read-only evidence
operations instead of selecting a fixed question capability. Enable it with
`COBOL_RAG_INVESTIGATION=1`; the default is disabled.

The model can list programs, query structured artifacts, filter a previous
collection, compare collections, inspect artifact fields, read source spans,
and search source text. Calls retain their caller, target, parameters and
COMMAREA in the same record. Tool failures are returned to the model so it can
repair its request within a bounded budget.

Session memory stores replayable collection recipes, their fingerprints, and
the displayed row identities. This distinguishes a full collection from its
displayed page and rejects stale results when the underlying corpus changes.

Answers are checked against requested formatting, evidence references, source
addresses and requirement coverage, then reviewed against the original
question. A reviewer failure cannot accept an unreviewed candidate. This review
uses the same configured model and is not an independent correctness proof.

The default budget is six model calls, ten tool calls and ninety seconds.
Italian adds up to two model calls for translation, using the existing
identifier-preserving translation checks. Output translation can fall back to
English; the debug data records that outcome. The configured HTTP timeout is
bounded by the remaining budget; it is not a hard process cancellation timer.

## Current boundaries

- Tools read existing analyzed artifacts; they do not regenerate analysis.
- Source search supports literal source lookup and scoped hybrid retrieval.
  Neither a limited page nor a ranked retrieval miss proves global absence.
- Inventory counts are counts of recorded analysis, not compiler-level proof.
- Graph nodes without incoming edges are not automatically proven dead code.
- Tool access is confined to analyzed program roots; no shell or writes are
  exposed to the model.
- New sessions are recommended when switching workflows.

The health endpoint reports `investigation_enabled`. In the platform, the
Compose configuration passes `COBOL_RAG_INVESTIGATION` into the API and pipeline.
Recreate the API after changing this environment variable. Preserve the model
and endpoint environment values used by your deployment when doing so.

Focused tests: `tests/test_investigation.py` covers collection memory, paging,
incoming call parameters, invalid filters, comparisons, and review failures.
Live model verification is also required before enabling a deployment.

## Local verification — 2026-09-14

Enabled locally with `gemma4:e4b-mlx`. Focused verification passed 61 tests
and 13 subtests. API checks confirmed a conversational greeting, PDB305's
recorded variable count of 105, and the first three flow-controlling variables
(DFHENTER, DFHPF1, DFHPF2). The final count check also verified that the session
retains PDB305 and the collection reference. These are smoke checks, not a
full evaluation of all question types or Italian translation.

During integration, equivalent flat tool envelopes were normalized, result
counts and references were preserved during context compaction, reviewer
exceptions were prevented from accepting candidates, and cited collection
references were attached to follow-up memory independently of model repetition.

## Verification and recovery — 2026-09-15

Kept `gemma4:e4b-mlx`; did not compare models or rebuild the corpus. A sequential
20-question run through `ChatSession.ask` scored 9 correct, 2 partial and 9
wrong/rejected against the actual artifacts. Targeted rechecks after repairs
give a rolling latest result of 14 correct, 2 partial and 4 failed. This is not
a new full-suite pass, and the targeted conversation omits some original turns.
Focused regression checks passed 98 tests and 19 subtests.

Saved repairs include typed entity mismatch diagnostics, source paragraph/COPY
context, an incoming-call primitive, natural-language support for verified empty
inventories, lossless source-span argument normalization, compact quality and
copybook-review fields, membership-first evidence previews, citation-lifetime
guidance, and review that permits genuine social conversation.

Known live gaps remain: 32 of 33 flow variables listed; stale collection handles
used as citations; compound unused-code requests missing quality evidence;
branch-condition requests answered with jump sites; dataflow claims missing the
exact cited edges; hybrid absence checks instead of exact identifier lookups.
The list-completeness annotation is currently optional and can be omitted by the
model. A same-model reviewer passing an answer is not correctness proof.

The disabled investigation flag still preserves the existing legacy workflow.
No changes here authorize writing source code or running shell commands through
the model's evidence tools. No claim is made that all live failures are repaired.
