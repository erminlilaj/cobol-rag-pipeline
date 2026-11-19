# Compact debug display and exports

Both the chat and Download + Debug now use the same compact renderer.

- Collapsed by default, with validation status, evidence count and trace label.
- Validation, call counts and recorded issues appear first when opened.
- A numbered execution timeline shows tool, status, returned count and recorded
  duration. Each step expands to its argument/details preview.
- Evidence is grouped by program/file rather than hundreds of top-level cards.
  Each group has a compact reference/use/source-preview table and cited count.
- Timelines, tables, warnings and raw data have bounded scrolling areas.
- Sources in exports are deduplicated by file and program and collapsed.
- One collapsed, escaped raw JSON block preserves complete original metadata,
  including fields not recognized by the renderer. Preview truncation is marked.
  This is visual compression, not deletion of diagnostics or a promise of a
  substantially smaller file. API-truncated evidence remains API-truncated.
- Missing validation is shown as not recorded, never as an assumed pass.
- Export format version is 2; no scripts or external assets are needed offline.
- Clean exports still do not serialize hidden debug metadata.

Verified: 21 JavaScript tests and 14 Python UI/progress tests passed. Tests include
250 evidence records, full metadata round-trip, escaping, unknown fields,
collapsed defaults, missing validation and source deduplication. The actual
19-Nov export was also re-rendered: four previously open debug panels become
four collapsed panels, with all diagnostic fields retained.

This changes UI/export only; it does not change answers, memory, models,
retrieval, analysis or the existing index. Pull the RAG test branch, restart
the API with start_rag.sh and refresh the browser. Existing downloaded HTML files
do not change retroactively; export again to get the new layout.
