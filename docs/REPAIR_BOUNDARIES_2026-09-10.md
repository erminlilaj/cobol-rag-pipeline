# Execution-boundary repair — 2026-09-10

## Scope

Incremental repair of the existing pipeline. No model, translation, index, analyzer,
or corpus changes. Existing uncommitted changes were preserved. No service restart
or fresh model-based evaluation was performed during this batch.

## Implemented

- Preserve a unique required subtask capability in typed compilation instead of
  losing it when no runtime config is supplied for reranking.
- Resolve call membership using call targets, not an overlapping copybook name.
- Preserve unresolved identifiers at the compiler boundary and stop unresolved
  named requests at the initial request boundary rather than comparing inventories.
- Distinguish missing/invalid membership artifacts from a valid empty inventory;
  retain the legacy variable-list artifact representation.
- Attach artifact sources and apply the existing response-contract validator to
  the post-refinement typed return path. This is not yet full typed fact validation.
- Validate coordinated identifier lists member by member, retaining the existing
  grounding requirements for each member. This narrowly handles copular identifier
  lists; it is not unrestricted prose decomposition or semantic entailment.

## Verification

Offline full suite: 481 passed, 1 skipped, 154 subtests passed.
Added 14 tests covering call/copybook name collisions, paraphrases, unknown names,
missing versus empty inventories, legacy artifacts, planned capability retention,
and multi-source lists with unsupported-member and unrelated-source decoys.

Replayed saved trace `8c046e77591341188505888cd811faea` using its saved evidence
excerpts and candidate answer. SYNCPOINT paragraphs RETURN-CICS, RETURN-MAIN,
ABEND00 now pass claim validation without citation repair or an LLM request.

These results establish component regressions, not a fresh end-to-end accuracy
score. The running API process has not been restarted to load this batch.

## Still required

- Replace lexical absence detection with typed claim validation; the selection
  not-found message false rejection is not fixed by this batch.
- Full evidence-result objects with applied-filter and coverage checks across all
  typed return paths; current source attachment reconstructs artifact provenance.
- Corpus-wide file inventory semantics.
- Declaration/initialization/increment and source/destination projections.
- COMMAREA option presence, paragraph bodies, and call preparation joins.
- Consistent main-source versus expanded-copybook call inventories.
- Fresh sequential end-to-end tests after deployment, including unseen paraphrases.

Do not equate a green unit suite with all reported questions being repaired.
