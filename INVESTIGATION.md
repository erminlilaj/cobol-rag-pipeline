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

## Evidence-contract repair — 2026-09-16 (live verification pending)

Known entity roles now reuse the existing scope catalogue. External calls and
copybooks cannot bypass evidence checks by labeling an answer general. Describe
preserves distinct roles, reports analysis gaps and includes source/expanded
CICS context for paragraph matches.

Call results preserve caller scope, target filters and direction, including
empty results and refined collections. Callees is the outgoing counterpart of
callers; the calls table no longer labels incoming lookups as outgoing. The
reviewer's interpretation of the original subject/direction is checked against
the executed relation. No question wording dispatches a tool.

Metrics carry explicit units and source artifacts. Answer checks reject known
value/unit swaps and unqualified paragraph counts. This is not a universal
prose fact checker: other paraphrases and cross-program attribution still rely
on semantic review, which uses the same model and remains fallible. Quality
counts explicitly identify commented-code records, not physical-line totals.

Verification: 52 focused offline tests passed. Five saved-corpus checks covered
metrics, external entity roles, call direction, expanded ABEND00 behavior and
quality-count units. No live model questions ran: Docker/API were unavailable.
No model, index, memory workflow or legacy answering-path changes were made.
Restart with the existing environment and verify the original questions plus
paraphrases before treating this as a verified live repair.

## Reviewer protocol repair — 2026-09-17

Live reproduction confirmed that outgoing, incoming and shared-call lookups
retrieved the right records, but the reviewer returned only passed/issues and
omitted the required relationship. Rewriting the answer could not repair this.

The reviewer now receives an explicit response schema, with direction-specific
caller/target arrays for multi-program requests. Its envelope is validated and
may be repaired once at the review stage within the existing call/time budget.
Repeated protocol failure stops rather than asking the answer writer to repair
the reviewer. Raw review JSON and protocol errors are recorded in the trace.
Direction checks still require matching evidence for every focal program.
Supplemental role metadata does not create an obligation on non-call queries.

Verification: 60 local contract tests passed. Focused live Gemma probes against
the configured runtime and saved corpus returned all five PDCBVC call targets,
no analyzed incoming callers of PDCBVC, shared PD0UTI01/PDPRED, 105 PDB305
variables, and PD1FS00's call/COPY roles. Intermediate probes exposed ambiguous
subject naming and over-strict metadata rejection; these were corrected before
the final checks. These were direct investigation-path probes, not browser E2E
tests or a broad accuracy benchmark. Memory, filtering, model and index unchanged.

## Variable evidence repair — 2026-09-18

Failed traces showed describe discarded access evidence, while nested previews
hid later accesses and the model repeatedly composed an unsupported data path.
Describe now retains variable evidence and relationships. A flat variable_access
table and typed tool expose pageable read/write/control sites from existing
artifacts; no source-specific routes or new inferred graph edges were added.
Singleton variable arguments normalize losslessly like program arguments.
Unknown source addresses remain available as unlocated_* analysis records,
not physical locations. Null fields are distinct from missing schema fields.
The model is instructed to inspect destination writes and source reads and the
reviewer to require actual check locations and exact data-flow endpoints.

62 offline tests passed. Focused live investigation probes returned the check
in LINK-PD1VOCI at 489 and WCTPAG -> COM3 -> NPG at 689/690. Intermediate runs
exposed argument-shape and unlocated-expression issues, corrected before the
final check. No broad benchmark, model change, or memory repair was performed.

## Follow-up memory repair — 2026-09-18

Six sequential ChatSession questions exposed a count follow-up changing from
five literal values to one variable after new queries replaced the active
collection during investigation. Previous-turn context is now snapshotted for
both generation and review. Active collections are fingerprint-checked and
recalled with fresh evidence IDs; recall does not mutate their recipes or focus.
Large collections recall counts only, never a misleading partial member list.
Memory exposes refined filters and units and strips old grouped citations.

65 offline tests passed. Final six-turn live sequence: 105 PDB305 variables;
all 18 flow-controlling variables; follow-up count 18; program switch preserving
the filter gives 33 for PDCBVC; five PDB305 WABEND-CODE literals; follow-up gives
5 but labels them write instances instead of literal values. Assessment: five
correct, one partial (semantic count-unit wording). Citation/grounding recovery
still consumes retries on some turns. This is not a general memory benchmark.

## Follow-up regression repair (supersedes automatic recall above)

Live API tests exposed failures not covered by the earlier scripted tests:
mandatory request metadata blocked valid tool decisions, count-only pages were
used to answer lists, wildcard variable access was mistaken for inventory,
and follow-up call targets lost their semantic role. Automatic evidence recall
is no longer used: previous context resolves references, while current tools
must supply evidence for the answer.

The investigation now resolves a standalone request before selecting tools,
preserves failed and successful conversational context, and distinguishes list
coverage from counts. Typed call tools support target sets and retain direction.
Invalid argument repair gets an isolated schema/error context and remains within
the existing call budget. An owning-program value incorrectly used as a
declaration-origin filter is rejected rather than accepted as evidence of zero
variables. The model must approve its suggested corrected query.

These changes retain the configured model and legacy non-investigation path.
They do not add question-specific answers or relax factual validation. JSON
transport remains enabled; local schema checks remain authoritative. A separate
context-resolution call adds overhead, so no latency improvement is claimed.

The resolver now receives the previous verified query (not just answer prose),
and review receives the original user question as its primary question. Corpus
scope cannot evade evidence requirements by being labeled general. Boolean and
unknown-field filters are validated against the unfiltered rows, so an invalid
predicate cannot become a false zero after an earlier predicate removes rows.

Focused verification: 88 offline tests passed. The final seven-turn live API
sequence passed: 105 variables; all 18 flow variables; the first five ordered
DFHENTER, DFHPF1, DFHPF2, DFHPF7, DFHPF8; all five PDCBVC callees; shared
PD0UTI01/PDPRED; WPD1AC with LENGTH(PD1AC-LUNGH); and no PDCBVC call to PD1AC.
Earlier live attempts failed and informed these repairs; this is one successful
final sequence, not a claim of universal or repeated-run reliability.

The additional seven-turn live smoke sequence also passed: available programs,
one-sentence purpose, source lines 207–215, comment-only follow-up, all twelve
PDB305 copybooks, a paraphrased flow-variable inventory (18), and its first three
alphabetically (DFHENTER, DFHPF1, DFHPF2). Final focused total: 14/14 answers
checked against the analyzed artifacts, not just reviewer acceptance. The API
was restarted with these changes; Gemma remained unchanged. Results are local,
uncommitted, and do not establish reliability for every unseen question.

## Explanation depth and debug visibility — 2026-09-22

Investigation responses now populate the UI evidence collection and expose the
resolved request plus execution steps, rather than showing a misleading empty
legacy plan. The UI distinguishes retrieved versus cited records and marks
truncated excerpts. This does not mean every retrieved record fit the LLM preview.

Depth and explanation goals are interpreted by the model. Detailed program
descriptions retrieve bounded samples of call interfaces, CFG edges, and CICS
operations alongside summary facts; the model still writes the answer. Samples
report coverage and are not an exhaustive program proof. Call records retain
types and relationship scope. Context previews preserve behavioral fields and
comparison subjects; retrieved records remain accessible in debug output.

Review checks depth, and local checks reject overview-only complete explanations
and unambiguous call-type/count contradictions. Optional null tool options are
omitted without relaxing required arguments. 93 offline tests passed. Live tests
confirmed one-sentence output, a richer PDB305 explanation with branching and
LINK/XCTL distinctions, and populated evidence (2 versus 23 records). Detailed
comparisons exposed further generation/review mistakes; do not equate acceptance
with full factual or explanatory completeness or claim these are universally fixed.
Final fresh comparison probe (trace 9cceb79cdaf54f2b82c48329ff6dee4f) failed to
retrieve evidence and was rejected. Detailed multi-program investigation remains
unresolved; the explanation-depth work is only partially verified live.
