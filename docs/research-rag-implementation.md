# Research RAG Implementation

This document records the implementation of the recommendations in the COBOL RAG research decision brief.

## Query Path

```text
question
  -> deterministic program/entity resolution
  -> reject ambiguous identifier prefixes
  -> inherit structured session state only for an explicit follow-up
  -> compile verified scope and literal constraints
  -> ask the LLM for a hierarchical plan (route, domain, tasks, relations, language)
  -> reject impossible conversational routes and retry with a compact LLM planner
  -> merge the semantic plan without dropping explicit user constraints
  -> execute supported tasks against authoritative structured artifacts
  -> retrieve a balanced evidence quota per program/entity when artifacts are insufficient
  -> exact-entity corrective retrieval when coverage is missing
  -> parent/sibling context expansion
  -> evidence guard
  -> cited answer or explicit abstention
  -> trace and optional user feedback
```

The LLM is the primary semantic query planner, not a COBOL fact source. Deterministic logic is limited to authoritative program/entity resolution, literal constraints, plan consistency, and schema-based evidence execution. A compact LLM planner is retried when the full plan is invalid; verified technical scope prevents an impossible conversational route.

## Implemented Research Recommendations

| Recommendation | Implementation |
|---|---|
| Hybrid routing before retrieval | Deterministic preprocessing resolves exact scope and literal constraints. The LLM returns route, category, domain, intent, tasks, relations, language, operations, exclusions, sources, and requested fields. Explicit constraints survive plan fusion; a contradictory conversational route is retried before bounded fallback. |
| Typed query planning | Every technical request becomes a `QueryPlan` containing domain, tasks, relations, response language, intent, program, multiple entities, operations/exclusions, evidence filters, division/section constraints, output fields, condition terms, and `result_scope`. Structured executors and retrieval consume the same contract. |
| Program/entity metadata filtering | Retrieval applies the resolved `program`; entity metadata and exact identifier coverage control reranking and validation. |
| Parent-child evidence | Records contain program, domain, and optional entity parent IDs. Retrieval selects task-specific child families, then adds bounded parent/entity/domain siblings; relation-rich requests receive a larger context allowance without admitting unrelated chunk types. |
| Structured session memory | `SessionState` stores the current program, a set of entities, intent, current query plan, last sources, and pending clarification. Only explicit follow-ups reuse entities; a new incompatible intent clears them. Assistant prose never becomes evidence. |
| Corrective retrieval | Missing exact-entity coverage triggers a lexical exact-identifier retry inside the selected program. |
| Evidence validation and recovery | Wrong-program, missing-evidence, and missing-exact-entity conditions prevent generation. The LLM emits structured claims. Missing citation syntax is repaired when a claim maps unambiguously to evidence; unsupported claims are dropped; one structured retry runs before the system abstains. |
| Per-answer tracing | JSON traces record route, scope, filters, vector/lexical/final hits, context roles, correction, guard, sources, and latency. |
| Corrective feedback | Feedback is stored by trace ID with rating, diagnostic labels, and a note. It does not automatically retrain the system. |
| Gold evaluation | `evals/pdcbvc_gold.jsonl` contains 47 cases covering authoritative facts, paraphrases, captured failed conversations, hierarchical paragraph/path tasks, call context, parent-child copybook usage, DB2/JCL separation, quality categories, pagination, exhaustive lists, stale-memory resets, multiple entities, qualifier contracts, ambiguity, out-of-scope routing, and injection-shaped input. |
| Prompt-injection controls | Retrieved text is delimited and declared untrusted; suspicious patterns are logged; unit tests verify the prompt boundary. |
| Index lineage and access metadata | Generated records include source/artifact hashes, schema/extractor version, hierarchy IDs, `access_scope`, and security classification. |

## Scaling Invariants

The system does not store a list of approved question wordings. Its reusable decision units are program, entity, intent, qualifier, evidence type, and citation support. Adding a new program should require new analyzed artifacts and a program manifest, not new answer strings.

- Exact identifiers resolve from the generated entity catalogue; multiple exact identifiers remain separate retrieval requirements.
- Deterministic handlers interpret common output contracts (for example, names only or source line/division/section) from the request and render fields from artifact schemas.
- The LLM planner handles paraphrases and decomposes requests into reusable task/relation fields without storing approved question strings.
- Comparisons retrieve and validate evidence independently for every requested program/entity so a large file cannot dominate the result.
- Unsupported identifiers and ambiguous prefixes produce clarification or abstention. Verified generated claims survive citation-format repair; unsupported claims never do.
- Confirmed production failures become stateful gold cases, so a fix cannot silently break an earlier conversation.

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
