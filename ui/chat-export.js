/* Self-contained, escaped HTML: readable offline, with no scripts or remote assets. */
(function (root) {
    const escape = value => String(value ?? '').replace(/[&<>"']/g, c =>
        ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
    function render({messages, sessionId, startedAt, exportedAt, timezone, includeDebug, pending = 0, running = false}) {
        const sections = messages.map((message, index) => {
            const sources = (message.sources || []).map(source => `<li>${escape(
                source.source_file || source.evidence_path || source.source_path || source.source_id || 'Source'
            )}${source.program ? ` — ${escape(source.program)}` : ''}</li>`).join('');
            const runtime = message.metadata?.runtime;
            const runtimeText = runtime ? `Model: ${runtime.llm}; embedding: ${runtime.embedding}; collection: ${runtime.collection}; server completed: ${runtime.completed_at}; duration: ${runtime.duration_ms} ms` : '';
            // Debug-free exports never serialize the metadata object.
            const debug = includeDebug && message.metadata
                ? `<details open><summary>Debug details — trace ${escape(message.metadata.trace_id || 'unavailable')}</summary><p>Intermediate candidates are not verified answers. Evidence may be excerpted by the API.</p><pre>${escape(JSON.stringify(message.metadata, null, 2))}</pre></details>` : '';
            return `<section><h2>${index + 1}. ${message.role === 'user' ? 'You' : 'AI'}</h2><p class="meta">${escape(message.timestamp)}</p><div class="answer">${escape(message.content)}</div>${sources ? `<h3>Sources</h3><ul>${sources}</ul>` : ''}${runtimeText ? `<p class="meta">${escape(runtimeText)}</p>` : ''}${debug}</section>`;
        }).join('\n');
        return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>COBOL RAG chat — ${escape(exportedAt)}</title><style>body{max-width:960px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui,sans-serif;color:#182230}h1,h2,h3{line-height:1.3}section{border-top:1px solid #ccd3db;padding:20px 0}.meta{color:#526070;font-size:13px;overflow-wrap:anywhere}.answer,pre{white-space:pre-wrap;overflow-wrap:anywhere}pre{background:#f2f4f7;padding:16px;font-size:12px}summary{cursor:pointer;font-weight:600}header{background:#f2f4f7;padding:20px}@media print{body{margin:0}details{display:block}}</style></head><body><header><h1>COBOL RAG — Chat transcript</h1><p>Exported: ${escape(exportedAt)} (UTC)<br>Local timezone: ${escape(timezone)}<br>Session: ${escape(sessionId)}<br>Browser transcript started: ${escape(startedAt)}<br>Messages: ${messages.length} · Debug details: ${includeDebug ? 'included' : 'excluded'}<br>Export format version: 1</p><p>This snapshot contains messages recorded in this browser tab since opening or clearing it, not earlier server history. ${running ? 'An answer was still running at export time. ' : ''}${pending ? `${pending} queued questions are not included. ` : ''}Unavailable runtime metadata is not inferred. Sources and debug information may contain private code; review before sharing.</p></header>${sections}</body></html>`;
    }
    root.ChatExport = {render};
    if (typeof module !== 'undefined') module.exports = {render};
})(typeof globalThis !== 'undefined' ? globalThis : window);
