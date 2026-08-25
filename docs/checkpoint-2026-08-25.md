# COBOL RAG checkpoint — 2026-08-25

## Objective

Build a scalable COBOL evidence assistant that can answer previously unseen questions across hundreds of analyzed programs. Answers must be scoped to the requested program, grounded in source-backed artifacts, complete for the requested contract, and explicit when the analysis does not contain the necessary evidence.

This checkpoint preserves the current architecture. It does not return to a fully deterministic question/answer system and does not replace the analyzer artifacts. The LLM remains responsible for semantic interpretation and answer composition; typed planning, exact evidence access, and validation constrain the parts where model guesses are unsafe.

## Architecture at this checkpoint

1. The platform runs the legacy analysis and cobol-rekt pipelines, combines their outputs, and publishes every completed program into one shared corpus.
2. A deterministic registry resolves explicit program and COBOL entity names before retrieval and prevents evidence from another program from leaking into an answer.
3. A hybrid query compiler combines typed English/COBOL parsing with an LLM semantic plan. It records intent, programs, entities, operations, qualifiers, requested fields, result scope, response format, and claim-level subtasks.
4. Direct artifact handlers answer exact, structured claims. Hybrid lexical/vector retrieval with exact-identifier boosting and bounded parent/child expansion supplies evidence for broader claims and joins.
5. Answer contracts enforce requests such as exact sentence/line counts, requested fields, exhaustive scope, exclusions, comparison coverage, and yes/no-first output.
6. Claim validation checks citations and requested-entity coverage. Repair and retry are bounded; rejected candidate answers remain available only in debugging details.
7. Structured session state supports explicit follow-ups without treating every short message as a continuation of an old technical question.
8. Per-answer traces expose the query plan, evidence inspected, attempts, disposition, and validation result.

## Improvements included

- Hybrid semantic routing instead of enumerating every possible question wording.
- Explicit program targeting and a multi-program corpus registry.
- Multi-entity and multi-source comparison support.
- Typed response contracts, claim decomposition, completeness checks, and bounded retries.
- Exact COBOL identifier resolution and metadata-filtered retrieval.
- Direct evidence handlers for variables, calls, CICS operations, copybooks, business rules, program metrics, forced values, control flow, and related artifacts.
- Typed evidence states that distinguish proven facts, proven absence, analysis gaps, retrieval misses, and rejected generated evidence.
- Normalized evidence views over the detailed analyzer artifacts, including stable entity keys, claim types, exact facts, and provenance.
- Richer variable analysis: declarations, parent/child groups, `REDEFINES`, read/write sites, read-write arithmetic, subscripts, SQL `SELECT INTO` writes, and flow-target filtering.
- Cleaner call evidence that preserves the real `CALL` source line and keeps preparation statements as pre-call context.
- Stale per-variable artifact cleanup so old analysis results cannot remain searchable after a rerun.
- Gold/probe evaluation growth, security and prompt-injection checks, observability, UI route labels, queue/stop behavior, and waiting games.

## Verification

- RAG unit/regression suite: **273 tests passed plus 24 subtests** in the deployed Python 3.12 container.
- Focused analysis suite: **17 tests passed**, covering normalized evidence, metadata, variable roles, and call-parameter extraction.
- Shared-corpus platform behavior: the new registry test passes. One older manifest-path test depends on host input paths that are not present in the isolated test-container layout and remains an environment-specific test limitation.
- The latest live development evaluation reported 41/41 development questions passing, but unseen manual questions revealed coverage gaps listed below. Development success must not be treated as sealed generalization evidence.

## What is working well

- Exact variable access, call inventory and parameters, CICS inventory, copybook inclusion, forced values, direct business-rule conditions, and many control-flow questions are reliable and source-backed.
- Successful direct-evidence answers are fast because the analyzer has already computed the facts; speed is not evidence that answers are hardcoded.
- The system generally rejects unsupported generated claims instead of fabricating them.
- The architecture can add programs incrementally without rebuilding or discarding programs already indexed.

## Known weaknesses

- Exact source-line lookup is not yet a first-class capability.
- Division/section range lookup, exact literal search, paragraph-body lookup, and some counterfactual path questions can be routed to broad semantic evidence instead of an exact source address.
- The LLM planner can still add unnecessary subtasks or omit a useful secondary claim, which creates latency and occasional completeness wobble.
- Citation repair remains safer than returning unsupported prose but can reject a correct candidate when the citation format, rather than the evidence, is wrong.
- Exhaustive and compound requests need continued sealed evaluation across artifacts and programs.
- Follow-up questions that change language or rely on vague pronouns remain less reliable than explicit English questions.
- The current evaluation is still dominated by PDCBVC and one additional sample; scale and isolation must be measured on a larger sealed multi-program corpus.

## Exact source-line lookup diagnosis

The inability to answer “what is on line 40 of PDCBVC?” is a missing foundational evidence-access path, not proof that the entire architecture is wrong.

The current planner can interpret `source line` as an output field—meaning “include line numbers in the answer”—but it does not consistently model a numeric source address as the entity being queried. Semantic retrieval is unsuitable for this operation: embeddings are designed to find meaning, not to guarantee a byte/line address. The raw COBOL source exists, but it is not currently exposed through a stable line-address contract.

The required capability is:

```text
source_line_lookup(
  program,
  source_file,
  line_start,
  line_end = line_start,
  context_before = 0,
  context_after = 0
)
```

The analyzer should publish a `program.source_lines.jsonl` or equivalent source-map artifact containing, for every physical line:

- program and source file;
- physical line number and original text;
- normalized statement text;
- division, section, and paragraph;
- comment/continuation status;
- provenance hash and analyzer version.

The registry should advertise a `source_line_lookup` capability. The query compiler should represent explicit numeric line/range references as typed source-address entities. Execution should read the exact indexed source record directly, optionally add neighboring lines, and bypass vector search and free-form LLM generation. The LLM may explain the returned code, but it must not select or invent the line.

The same address layer should power exact section boundaries, paragraph bodies, literal occurrences, and “show only lines X–Y.” This is a general capability addition and should be evaluated with unseen line numbers and multiple programs; it must not be implemented as PDCBVC-specific wording or answers.

## Next quality gate

Before broader feature work, implement exact source-address artifacts and handlers, then run a sealed suite containing:

- unseen single-line and line-range lookups;
- section and paragraph boundary questions;
- exact literal occurrences;
- the same addresses in at least two programs;
- ambiguous requests that omit the program in a multi-program corpus;
- follow-ups such as “show five lines around it” without losing the resolved address.

Success means exact text and metadata match the source, no cross-program evidence is returned, requested context size is honored, and the answer is reproducible without relying on embedding similarity.
