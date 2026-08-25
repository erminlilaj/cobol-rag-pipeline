# Research RAG Implementation

This document records the implementation of the recommendations in the COBOL RAG research decision brief.

## Query Path

```text
question
  -> deterministic program/entity resolution
  -> reject ambiguous identifier prefixes
  -> inherit structured session state only for an explicit follow-up
  -> compile an authoritative QueryPlan and response contract
  -> ask the LLM for a hierarchical plan (route, domain, tasks, relations, language)
  -> reject semantic additions that were not authorized by the typed plan
  -> choose the least expensive execution lane
  -> execute supported tasks against authoritative structured artifacts
  -> retrieve a balanced evidence quota per program/entity when artifacts are insufficient
  -> exact-entity corrective retrieval when coverage is missing
  -> parent/sibling context expansion
  -> evidence guard
  -> cited answer or explicit abstention
  -> trace and optional user feedback
```

The LLM is a semantic planner and presentation renderer, not a COBOL fact source. Deterministic logic is limited to authoritative program/entity resolution, typed claims, literal constraints, plan consistency, and schema-based evidence execution. A compact LLM planner is retried when the full plan is invalid; verified technical scope prevents an impossible conversational route.

## Adaptive Execution Lanes

The system does not send every request through the most expensive path.

1. **Conversational/general knowledge** — no COBOL retrieval and no program ambiguity check.
2. **Direct structured evidence** — exact program/entity questions and same-capability comparisons read canonical artifacts.
3. **Standard grounded RAG** — one evidence capability needs retrieval and generation.
4. **Bounded agentic execution** — only multiple evidence capabilities or multiple programs are split into independently validated claims.

Agentic execution has a bounded number of capability retries. It cannot walk the entire capability catalogue, expand the requested scope, or change the response contract.

## Authority Boundary

For a high-confidence technical request, the initial typed `QueryPlan` is immutable authority. A semantic planner proposal may improve descriptions and split authorized tasks, but it may not add or replace:

- programs or exact COBOL entities;
- tasks or evidence capabilities;
- relations, output fields, exclusions, or comparison scope;
- response language or presentation constraints.

Rejected semantic additions are recorded in `plan.policy_rejections` and therefore appear in the answer trace and debugging UI. A low-confidence `general` plan may still be resolved semantically; after resolution it is normalized through the same typed capability schema.

The response contract remains attached to every subplan. Evidence-stage validation deliberately ignores presentation limits such as `max_lines`, while the final renderer must satisfy those limits without adding claims. Natural requests such as “in three lines” compile to an exact line count; explicit wording such as “at most three lines” remains an upper bound. If the bounded renderer cannot produce a grounded compliant answer, the candidate and rejection reasons remain visible in debugging rather than being returned as trusted output.

## Typed Evidence Boundary

Retrieval results are normalized into evidence records with an evidence ID, program, evidence type, source file, entity key, score, and metadata. Debug output classifies the result as one of:

- `proven`;
- `proven_absent`;
- `analysis_gap` (the analyzer did not produce the required capability);
- `retrieval_miss` (the capability exists, but no matching record was retrieved);
- `evidence_rejected` (records were retrieved but failed program/entity/claim validation);
- `not_applicable` for conversation.

This distinction tells us whether to improve the analysis pipeline, retrieval, routing, or answer validation instead of treating every failure as “no evidence.”

## Implemented Research Recommendations

| Recommendation | Implementation |
|---|---|
| Hybrid routing before retrieval | Deterministic preprocessing resolves exact scope and literal constraints. The LLM returns route, category, domain, intent, tasks, relations, language, operations, exclusions, sources, and requested fields. Explicit constraints survive plan fusion; a contradictory conversational route is retried before bounded fallback. |
| Typed query planning | Every technical request becomes a `QueryPlan` containing domain, tasks, relations, response language, intent, program, multiple entities, operations/exclusions, evidence filters, division/section constraints, output fields, condition terms, and `result_scope`. Structured executors and retrieval consume the same contract. |
| Immutable plan authority | High-confidence typed scope cannot be widened by the LLM. Rejected task/capability/field additions are traced as `policy_rejections`. |
| Adaptive execution | Same-capability work stays direct; only cross-capability or cross-program requests use bounded claim decomposition. |
| Program/entity metadata filtering | Retrieval applies the resolved `program`; entity metadata and exact identifier coverage control reranking and validation. |
| Parent-child evidence | Records contain program, domain, and optional entity parent IDs. Retrieval selects task-specific child families, then adds bounded parent/entity/domain siblings; relation-rich requests receive a larger context allowance without admitting unrelated chunk types. |
| Structured session memory | `SessionState` stores the current program, a set of entities, intent, current query plan, last sources, and pending clarification. Only explicit follow-ups reuse entities; a new incompatible intent clears them. Assistant prose never becomes evidence. |
| Corrective retrieval | Missing exact-entity coverage triggers a lexical exact-identifier retry inside the selected program. |
| Evidence validation and recovery | Wrong-program, missing-evidence, and missing-exact-entity conditions prevent generation. The LLM emits structured claims. Missing citation syntax is repaired when a claim maps unambiguously to evidence; unsupported claims are dropped; one structured retry runs before the system abstains. |
| Per-answer tracing | JSON traces record route, scope, filters, vector/lexical/final hits, context roles, correction, guard, sources, and latency. |
| Corrective feedback | Feedback is stored by trace ID with rating, diagnostic labels, and a note. It does not automatically retrain the system. |
| Gold evaluation | `evals/pdcbvc_gold.jsonl` contains 74 development cases covering authoritative facts, paraphrases, captured failed conversations, hierarchical paragraph/path tasks, call context, parent-child copybook usage, DB2/JCL separation, quality categories, pagination, exhaustive lists, stale-memory resets, multiple entities, qualifier contracts, ambiguity, out-of-scope routing, and injection-shaped input. |
| Contract and isolation evaluation | Accepted technical answers are automatically checked against their response contract, and every cited source must belong to the resolved program. Reports include pass rates per evaluation category. |
| Prompt-injection controls | Retrieved text is delimited and declared untrusted; suspicious patterns are logged; unit tests verify the prompt boundary. |
| Index lineage and access metadata | Generated records include source/artifact hashes, schema/extractor version, hierarchy IDs, `access_scope`, and security classification. |

## Scaling Invariants

The system does not store a list of approved question wordings. Its reusable decision units are program, entity, intent, qualifier, evidence type, and citation support. Adding a new program should require new analyzed artifacts and a program manifest, not new answer strings.

- Exact identifiers resolve from the generated entity catalogue; multiple exact identifiers remain separate retrieval requirements.
- Deterministic handlers interpret common output contracts (for example, names only or source line/division/section) from the request and render fields from artifact schemas.
- The LLM planner handles paraphrases and decomposes requests into reusable task/relation fields without storing approved question strings.
- Comparisons retrieve and validate evidence independently for every requested program/entity so a large file cannot dominate the result.
- Lexical corpus snapshots are partitioned by program and compiled once per cache window, avoiding a full-corpus read and tokenization pass for every query.
- Unsupported identifiers and ambiguous prefixes produce clarification or abstention. Verified generated claims survive citation-format repair; unsupported claims never do.
- Confirmed production failures become stateful gold cases, so a fix cannot silently break an earlier conversation.

## Analyzer-To-RAG Evidence Contract

The analyzer remains the source of truth, but retrieval no longer has to learn every
analyzer-specific JSON shape. Each run also emits an additive
`evidence.normalized/evidence.normalized.json` view with a stable schema:

```text
program + capability + entity type/key + claim type
summary + typed facts(role, paragraph, line, source file, statement)
attributes + source-artifact provenance
```

Detailed artifacts are preserved for deterministic execution and debugging. The
normalized view is indexed as the primary semantic representation, so the LLM sees
compact, typed facts while citations still resolve to the original artifacts.
Variable views are restricted to the canonical `dataflow.used_variables.json`
inventory. Reused runs mirror analyzer-owned directories and remove obsolete
per-variable files, preventing a corrected false variable from remaining searchable.

Analysis and RAG failures are diagnosed separately. A missing capability is an
`analysis_gap`; a present capability with no matching hit is a `retrieval_miss`; an
unsupported generated claim is `evidence_rejected`. This allows improvements to be
made in the analyzer when evidence is absent instead of adding question-specific
router patterns.

## Verified Plan Compiler And Corrective Contracts

Typed parsing owns literal constraints such as program, COBOL identifiers, COPY/CICS
keywords, requested fields, exclusions, counts, and exact lines. Semantic routing may
classify and decompose the request, but its confidence cannot grant deterministic
authority or widen a high-confidence typed plan. Weak semantic plans are recompiled
to one evidence capability rather than accumulating unrelated subtasks.

When corrective execution substitutes a capability, final validation uses the tasks
of the successful evidence path while preserving the original program, entities,
requested fields, and presentation contract. This prevents a verified variable test
from being rejected merely because a business-rule artifact was unavailable. COBOL
syntax words such as `USING`, `INTO`, and `SELECT` are typed qualifiers, not corpus
entities. An answerability gate runs only for low-confidence technical requests and
asks for clarification before retrieval when no coherent evidence capability exists.

## Record Metadata

The analysis factory emits these retrieval and lineage fields when applicable:

```text
program
intent_domain
entity_type / entity_key
variable / paragraph / target / copybook
hierarchy_level
parent_id / parent_type
program_parent_id
domain_parent_id
entity_parent_id
source_file / evidence_path
source_sha256 / artifact_hash / content_hash
extractor_version / index_schema_version
access_scope / security_classification
```

`access_scope` is present for future authorization enforcement. The current local single-user API records it but does not implement enterprise authentication or row-level access control.

## Context Roles

Retrieved context is ordered as:

1. `exact_child`
2. `retrieved_child`
3. `parent_context`
4. `entity_sibling`
5. `domain_sibling`

Expansion is deliberately bounded to avoid replacing precise evidence with broad context.

## Trace And Feedback API

Retrieve a trace:

```bash
curl http://localhost:8000/api/traces/TRACE_ID
```

Attach feedback:

```bash
curl -X POST http://localhost:8000/api/feedback \
  -H 'Content-Type: application/json' \
  -d '{
    "trace_id": "TRACE_ID",
    "rating": "incorrect",
    "labels": ["wrong_entity"],
    "note": "The retrieved variable was not the requested one."
  }'
```

Allowed ratings are `correct`, `partial`, and `incorrect`. Diagnostic labels include `wrong_program`, `wrong_entity`, `missing_identifier`, `bad_chunking`, `weak_rerank`, `missing_parent`, `hallucination`, `failed_abstention`, `missing_evidence`, and `other`.

## Evaluation

Run:

```bash
python -m cobol_rag.evaluation \
  --config /path/to/runtime.yaml \
  --gold evals/pdcbvc_gold.jsonl \
  --final-scripts-dir /path/to/final_scripts/PROGRAM \
  --output-dir /path/to/eval-output
```

The runner writes JSON details and a Markdown summary, including expected `QueryPlan` fields for qualifier-sensitive cases. Add every confirmed production failure to the JSONL suite before changing prompts, routing, chunking, or reranking.

## Security Boundary

- Retrieved text is data, never an instruction source.
- Evidence records are explicitly delimited in the generation prompt.
- Exact entity questions must have exact entity coverage.
- The local RAG service is read-only with respect to COBOL source repositories.
- Traces and feedback are local operational data and may contain user questions; protect the runtime directory appropriately.

## Current Deliberate Limits

- Session state is in-process and intended for the current single-user local UI.
- Feedback is offline diagnostic data; there is no automatic self-training.
- Parent IDs are logical hierarchy links, not a graph database. Bounded CFG path traversal is implemented over retrieved control-flow evidence; unrestricted graph-wide synthesis is not.
- Balanced cross-program comparison routing is implemented; enterprise authentication, automatic analyzer-conflict adjudication, and persistent distributed session state remain future work.
- The local 8B planner can be slow and may produce invalid labels or brittle multilingual cited output. The system retries routing, repairs supported claims, and abstains rather than weakening evidence validation; route-plan caching and a smaller dedicated routing model remain performance improvements.
