# Semantic query and variable repair — 12 September 2026

## What changed

- The primary planner uses schema-constrained JSON with separate system/user messages. Technical responses must contain an executable specification. A single bounded repair handles invalid planning; known evidence requests cannot silently become conversational acknowledgements.
- Variable inventory execution now supports count, filtering, ordering, offset and limit. Variable detail execution reads declarations, PIC, origin, group relationships and requested access sites.
- Session memory stores the ordered selected variable results. Ordinal references resolve against that result, not the original input entities. The planner receives a bounded preview instead of serializing the full inventory into its context.
- Executable specifications take precedence over keyword-derived tasks. Validation of a variable type no longer demands unrelated read/write/comparison sections.
- Incoming/outgoing call direction is preserved by the compiler. Missing direction requires semantic repair instead of another English-wording inference. Call parameter/preparation and filtered count execution use call artifacts.
- Source addresses and corpus inventory go through semantic selection; exact readers still provide the evidence. Identifier components such as MAP inside PREPARA-MAP-010 are not treated as independent inventory requests.
- Capability availability is checked after selection, avoiding a preliminary embedding-ranking pass for simple counts. Context annotations are no longer reparsed as part of the original question by the structured-answer fallback.

## Verification

Focused tests: 32 passed, plus 13 unittest subtests. Two older tests were updated because they expected the compiler to infer missing call direction from English; the new boundary requires the semantic planner to supply it.

Final live sequence using Granite 4.2 8B and the existing corpus:

1. How many variables does PDB305 have? — 105.
2. List the first three of them only. — ASKIP-DRK, ASKIP-DRK-FSET, COM3.
3. What type is the first one? — ASKIP-DRK, elementary item, PIC X, WORKING-STORAGE, source line 99.

All three final answers passed validation. The live planner still needed its bounded repair on some turns; this is not a claim of perfect first-pass planning.

Additional checks: the live filtered inventory returned 18 flow-controlling PDB305 variables. Artifact-only execution returned ABEND00 statements, WPD1AC for the PD1AC call, and the actual text at PDCBVC line 50.

Early staged candidates failed with context overflow, invalid schema choices, and overly broad validation requirements. Those failures were found before installation. No full evaluation suite or exhaustive Italian/cross-program regression run was performed. Broad synthesis and complex compound queries still need user evaluation; not every historical failure is claimed fixed.

No input files, analysis artifacts, index, embedding model or generation model were changed. Pre-install source copies are in `/private/tmp/cobol-rag-before-repair.TTbQte` (temporary backup; not a Git checkpoint). No commit or push was made.
