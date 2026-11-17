# Bounded indexing repair — 17 November 2026

## Evidence and scope

Prepared against deployed RAG revision `9fa4beb`, without changing model,
investigation behavior, memory settings or analyzer outputs. PDHVAL10 was killed
while applying changes; PDHVAL11 applied 2,371 changes in about 105 minutes and
was then killed at manifest saving. The manifest was reported as 584 MB. The API
was using about 7.8 GiB of Docker's 19.5 GiB allocation before it was stopped.
The supplied Docker event query returned no OOM event. These facts justify
addressing memory/IO scaling, but do not prove the cause of either SIGKILL.

## Changes

- Stream the existing JSON manifest with `ijson`; retain only incoming IDs and
  obsolete-source candidates within the current synchronization root.
- Cache repeated path-containment checks within one plan. Do not change path
  resolution semantics or assume that an outside source is safe to delete.
- Skip vector opening and manifest rewriting for genuinely unchanged plans.
  Treat source-path/format changes as updates so provenance is not silently stale.
- Stream merged JSON into a same-directory temporary file, flush/fsync the file,
  then atomically replace the manifest. Preserve unrelated sources and the BM25
  reference; keep the old manifest if serialization or writing fails.
- Replace pre-chunked vectors in batches of at most 32. Use one source-ID metadata
  filter per batch rather than one filtered deletion per document. This uses
  Chroma's documented deletion/filter API:
  https://github.com/chroma-core/chroma/blob/main/chromadb/api/models/Collection.py
- Calculate embeddings before deleting prior vectors. An embedding-service
  failure must not remove otherwise usable vectors.
- Still clean up additions: a previous interrupted process may have inserted
  their vectors without committing the manifest. Preserve single-document
  behavior for repeated identities or non-pre-chunked content.
- Write a small per-sync-target pending journal before modifying vectors.
  For the same target, retry touched sources even if their current hash matches
  the old manifest; remove journaled orphan additions when no longer in inputs.
  Clear the journal only after successful manifest saving.

This is not a multi-writer transaction or a resume-from-the-last-batch scheduler.
Keep index writers serialized. A retry can repeat work, but must converge rather
than trust a stale manifest. New journals cannot reconstruct undocumented
partial writes from earlier versions automatically.

## Dependency and deployment

`ijson>=3.3,<4` is added to both the RAG package and the platform Docker RAG
requirements. The platform image needs rebuilding; merely restarting the API
does not install the new dependency. No index format migration, vector-store
reset or blanket analyzer rerun is required by this repair.

Keep the API stopped for the first retry to reduce memory contention. Preserve
the existing artifacts. Test PDHVAL10 alone before resuming the dataset, inspect
the final report and resource usage, then verify retrieval and a cached retry.
Do not infer success solely from a 100% vector-change counter: the manifest and
overall stage must complete too.

## Local verification and limits

Tests include real disposable Chroma collections with deterministic stubbed
embedding transport (no live Ollama or production index): batch counts,
unchanged-document skips, failed manifest recovery, reverted-input repair,
orphan cleanup, failed embeddings preserving old vectors, metadata/BM25
preservation, invalid-manifest rejection, atomic-write failure and bounded
memory on 20,000 synthetic manifest entries. A separate existing investigation,
incoming-call and corpus-reference regression group was also exercised.
Combined result: **211 tests passed, plus 20 passing subtests**. Python compilation
and diff whitespace checks passed. The synthetic memory assertion stays below
5 MiB of Python-traced allocations while filtering/rewriting 20,000 entries;
this is not a measurement of production Chroma RSS.

The 584 MB remote manifest and the production Chroma index were not copied or
benchmarked locally. Docker image construction and live PDHVAL10 completion
remain deployment checks. The observed API/block-IO totals do not independently
identify current disk throughput or prove a particular memory leak.
