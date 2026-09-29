# Investigation repair — 2026-09-27

## Scope

Repairs to the existing model-led investigation path. The model, corpus,
embedding model, memory setting and legacy query implementation were not changed.
Existing uncommitted work was preserved.

## Changes

- Retain request offset/order and validate the requested page, not an oversized inventory.
- Keep stable result handles after context reduction; copy observation previews before trimming.
- Compact comparison members rather than duplicating nested parameter payloads.
- Expose bounded source windows for variable-access records, including condition continuations.
- Join available copybook-review facts to inclusion records, with provenance and unknown status
  when review analysis is absent. Inclusion alone does not prove unused status.
- Preserve source PERFORM/GO TO/CALL separately from generic CFG edge categories.
- Annotate source declaration syntax, including commented group declarations without PIC.
- Explain the proof scope of empty access and call results explicitly.
- Repair review formatting in isolation; do not force new retrieval for every prose rejection.
- Preserve typed caller/target contracts, including equivalent singular incoming target notation.
- Re-review scoped empty relations using their empty attribute projections; no automatic approval.
- Normalize equivalent protocol predicates and optional null fields without inventing evidence.
- Increase JSON answer allowance from 1,000 to 2,048 tokens and budget the final repair envelope.

## Verification

- Investigation and typed-query suites: **153 passed, 21 subtests passed**.
- Real artifact checks: shared calls PD0UTI01/PDPRED; 18 flow-controlling variables;
  correct sixth–tenth page; DFHBMSCA/PDRTELR review candidates; condition continuation
  retained; PERFORM distinguished from graph CALL.
- Targeted live Gemma checks ultimately passed: available programs, shared calls,
  and incoming callers/parameters for PDCBVC. These are not a full transcript rerun.
- Legacy final-script suite: **57 passed, 7 failed**. The same seven failures recur with
  the three edited investigation modules blocked from import, so they are independent
  of execution through this repaired path. Legacy behavior was not modified to silence them.

## Remaining verification

The entire 61-attempt transcript has not been rerun. In particular, detailed explanations,
quality subset selection, negative declaration questions and paraphrase reliability need
fresh end-to-end checks. Passing offline contracts does not prove every generated answer
correct. The legacy suite's seven failures remain open.

## Subsequent regression repair (same day)

The first repair was insufficient. The later user transcript exposed a destructive
context fallback: executable/source facts were replaced by identifiers while the reviewer
could still see other evidence. Earlier passing checks did not cover that final envelope.

Changes in this round:

- Remove metadata-only fallback and shorten repeated instructions. Retain complete small
  results and actual statements in larger previews. Whole older previews can be evicted,
  with explicit result handles and an omission notice; overflow never authorizes fabrication.
- Freeze the user's request: reviewer suggestions are diagnostic, not replacement requests.
- Normalize equivalent ordering and tool envelopes. Retain copybook predicates in the wrapper.
- Preserve call-direction provenance through set operations. Add `compare_groups` for generic
  comparisons inside one collection, avoiding fragile chains of intermediate model-generated IDs.
- Include entry/exit edges with paragraph evidence. Add bounded `data_dependencies` joins over
  existing read/write sites. These are static may-dependencies, not runtime feasibility proofs.
- Non-count requests receive detail rows rather than count-only access/edge results. Rejected
  evidence/interpretation triggers retrieval rather than repeated rewrites of unsupported prose.
- Reject citation-only source-line entries when the cited line contains nonblank text.
- Investigation defaults: 16 model calls plus the existing two-call recovery reserve,
  20 top-level tool calls, 240-second deadline, 16,384-token context. These are ceilings,
  not calls consumed for every question. The current model advertised 131,072 context tokens.

Model remains `gemma4:e4b-mlx`; memory remains disabled. No corpus files were rewritten.
Focused offline verification: **162 passed, 21 subtests passed**. Direct artifact verification
finds the two-statement WCTPAG/COM3/NPG path and all six incoming ABEND00 graph edges.
Live checks during development exposed additional protocol and subset-validation failures;
their earlier rejections must not be counted as successful tests. Final live coverage is
reported separately below; this is not a claim that every unseen question is solved.

## Resumed verification — 2026-09-28

- API restarted with `gemma4:e4b-mlx`; health endpoint reports OK; memory disabled.
- **163 tests passed, 21 subtests passed** in the investigation/query-IR suites.
- Model tool batches allow up to six requests, still within the global tool budget.
  Whole-paragraph span aliases are normalized without inventing numeric source addresses.
- Reviewer paragraph context is restricted to explicitly selected programs, avoiding
  accidental cross-program joins for identically named paragraphs.
- Live entry-condition check now lists all six recorded PDCBVC ABEND00 conditions and
  locations, distinguishing PERFORM from GO TO (trace `1ae50fab6ba840df8489039ef839e522`).
- Live indirect-dataflow check now explains WCTPAG -> COM3 -> NPG without the earlier
  unsupported COMMAREA label (trace `0303416cbf604dffaaae01749688aa51`).
- Earlier successful live checks covered all 18 flow-controlling variables, alphabetical
  first-five selection, copybook review candidates, literal source excerpts, shared calls,
  a simple program explanation and call parameters. These were development checks, not
  a fresh full 40-question evaluation of the final configuration.

## Publication checkpoint — 2026-09-29

- Corrected paragraph evidence boundaries: trailing comments no longer imply
  behavior of the preceding paragraph; physical source text remains accessible.
- Inclusion and listing directives have explicit roles. Paragraph-body views
  omit listing controls; exact source-line requests still return them.
- Expanded CICS operations and reviewer execution contracts distinguish static
  graph edges, termination, COMMAREA and LENGTH semantics.
- Equivalent nested reviewer repair envelopes are normalized without changing
  rejection verdicts. Retrieved inline citations are reconciled with metadata;
  unknown evidence references still fail validation.
- Added timestamped HTML chat export with optional debug details and API runtime
  metadata. Export contents are escaped and require no external assets.
- Publication verification: 140 focused investigation/UI tests and 3 JavaScript
  chat-export tests passed. These are not 143 live answer-quality tests.
- Five focused live requests all returned answers, including an exact repeat.
  Brief explanations still omitted ROLLBACK, and the LINK-PD1AC check still
  misexplained LENGTH. Validator acceptance is not a correctness guarantee.
- Memory stays disabled; model stays gemma4:e4b-mlx. No local corpus, model
  weights, environment files, logs or generated package metadata are published.
