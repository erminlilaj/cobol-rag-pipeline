# Compound quality-query repair — 10 September 2026

## Scope

Issue 5 only: requests combining unused/dead code, commented-out code,
unreachable paragraphs, and unused/review copybooks. No model, translation,
embedding, index, source-analysis, or other capability changes were made.

## Root cause

The planner added code tasks only when its task list was empty. A copybook
request populated that list first, suppressing the code part of a compound
question. The typed-query fallback independently used early returns, so it
could also lose categories and interpreted hyphenated wording differently.
The legacy direct-artifact route had a separate commented-code renderer.
Finally, citing the quality artifacts did not prove that every requested
category appeared in the answer.

## Repair

- `quality_scope.py` centralizes the additive quality taxonomy for text
  fallback and required semantic claim tasks. Semantic obligations can execute
  without matching a particular question phrase.
- Planning and compilation use the same categories. Required claim-level
  tasks survive into execution; optional claims do not expand the scope.
- The direct-artifact and typed-query routes use the same category-aware
  quality renderer.
- Every requested category reports findings, an explicitly empty recorded
  result, or unavailable analysis. Missing records are not treated as proof
  of absence.
- Compound prose validation checks category coverage, excluding source and
  scope footers. This is a coverage check, not a proof of reachability or
  compiler-level non-use; count-only and JSON presentation remain separate.

## Verification

Full suite: **506 passed, 1 skipped, 154 subtests passed**.

The 25 new regression cases cover both real program names and an unrelated
synthetic program, reordered and hyphenated compound requests, narrow
single-category requests, required versus optional claims, a narrowed LLM
query specification, incomplete answers with misleadingly broad citations,
and empty versus unavailable category records.

The API was restarted without rebuilding or reindexing. Live checks use
fresh sessions and the existing Granite 4.2 model. Live results are recorded
below after completion.

## Evidence boundaries

The current artifacts report these values:

| Program | CFG-reported candidates | Commented-out entries | Proven unused copybooks | Copybooks needing review |
| --- | ---: | ---: | ---: | --- |
| PDB305 | 4 | 68 | 0 | DFHBMSCA, PDRTELR |
| PDCBVC | 5 | 15 | 0 | DFHBMSCA, PDRTIP01, PDRVC |

These graph-based candidates are not compiler-expanded dead-code proof.
Review copybooks must not be described as proven unused. Default answers
still show the first eight commented-out entries and explicitly state the
remaining count; they are summaries, not exhaustive code listings.
