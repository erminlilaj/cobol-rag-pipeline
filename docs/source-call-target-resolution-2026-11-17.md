# Source-backed variable CALL targets

The published call inventory can preserve a variable operand such as WNOMEPGM
as its target. That is not necessarily the concrete program invoked. For example,
`MOVE 'PD2DF' TO WNOMEPGM` immediately before `CALL WNOMEPGM USING WPD1DF`
supports a call-site target of PD2DF, with WPD1DF as its argument.

`call_targets.py` adds a conservative read-time enrichment over existing source
maps and call records. It verifies the exact source member, paragraph and call
location, then accepts an immediately preceding literal MOVE in straight-line
source. It preserves the variable operand, parameters, call type and assignment /
call provenance. It does not infer targets from paragraph names or field names.

Conditional/control-flow-bearing paragraphs, missing source lines, unknown
intervening operations, complex operands and unsupported continuations remain
unresolved. This is intentionally not general interprocedural constant propagation.

The resolved record is used before callers/callees/parameter preparation filters,
and in summaries and corpus graph query views. Stored analyzer artifacts and the
published graph are not rewritten: read-time evidence retains their recorded
target and adds source-backed resolution. Global graph queries involving variable
targets may read source for multiple programs; kind/source filters narrow that work.

Model and reviewer instructions distinguish the called program from the variable
holding its name. Resolution proof remains in bounded evidence previews. A known
resolved target does not prove its implementation has been analyzed.

No COBOL analysis rerun or embedding rebuild is required for this read-time fix
when the needed source maps already exist. Deploy the RAG code and restart the
API. If source is genuinely missing or stale, repair that analysis separately.
Memory settings and the selected model are unchanged.
