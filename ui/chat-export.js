/* Self-contained, escaped HTML: readable offline, with no scripts or remote assets. */
(function (root) {
    const escape = value => String(value ?? '').replace(/[&<>"']/g, c =>
        ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
    function runtimeSummary(runtime) {
        const duration = runtime?.duration_ms;
        const time = typeof duration === 'number' && Number.isFinite(duration) && duration >= 0
            ? (duration < 60000 ? `${(duration / 1000).toFixed(1)}s`
                : `${Math.floor(duration / 60000)}m ${((duration % 60000) / 1000).toFixed(1)}s`)
            : 'unavailable';
        const usage = runtime?.token_usage;
        const count = value => Number.isSafeInteger(value) && value >= 0;
        let tokens = 'Tokens: unavailable';
        if (usage && count(usage.input_tokens) && count(usage.output_tokens)
            && count(usage.total_tokens) && usage.total_tokens === usage.input_tokens + usage.output_tokens) {
            const format = value => value.toLocaleString('en-US');
            const complete = usage.complete === true && usage.reported_calls === usage.model_calls;
            tokens = `Tokens${complete ? '' : ' (reported subtotal)'}: ${format(usage.total_tokens)} `
                + `(in ${format(usage.input_tokens)} / out ${format(usage.output_tokens)})`;
            if (!complete) tokens += ` · ${usage.reported_calls ?? '?'} / ${usage.model_calls ?? '?'} calls reported`;
        }
        return `Time: ${time} · ${tokens}`;
    }
    const list = value => Array.isArray(value) ? value : [];
    const short = (value, size = 180) => {
        const text = String(value ?? '');
        return text.length > size ? text.slice(0, size) + '…' : text;
    };
    const fold = (title, body, extra = '') => `<details class="debug-nested ${extra}"><summary>${escape(title)}</summary>${body}</details>`;
    function evidencePreview(item) {
        let value;
        try { value = JSON.parse(item.excerpt); } catch { /* API may already truncate excerpts. */ }
        const text = value?.text || value?.statement || value?.name || value?.paragraph || item.excerpt || 'No excerpt recorded.';
        return `${value?.line != null ? `L${value.line}: ` : ''}${short(text, 140)}`;
    }
    function debugView(debug, traceId = '', raw = debug) {
        if (!debug || !Object.keys(debug).length) return '';
        const validation = debug.validation || {};
        const investigation = debug.investigation || {};
        const steps = list(investigation.steps).length ? investigation.steps : list(debug.plan?.steps);
        const evidence = list(debug.retrieval?.evidence);
        const passed = validation.passed === true ? 'Passed' : validation.passed === false ? 'Failed' : 'Not recorded';
        const status = debug.status || passed;
        const issues = [...list(validation.reasons)];
        for (const step of steps) {
            issues.push(...list(step.validation_errors), ...list(step.review_protocol_errors));
            if (step.error) issues.push(step.error);
            if (step.resolution_error) issues.push(step.resolution_error);
            if (step.review_response?.passed === false) issues.push(...list(step.review_response.issues));
        }
        const uniqueIssues = [...new Set(issues.map(value => typeof value === 'string' ? value : JSON.stringify(value)))];
        const groups = new Map();
        for (const item of evidence) {
            const file = item.source_file || item.evidence_path || item.source_id || 'Unspecified source';
            const label = [item.program, file].filter(Boolean).join(' · ');
            if (!groups.has(label)) groups.set(label, []);
            groups.get(label).push(item);
        }
        const facts = [
            ['Validation', passed], ['Stage', validation.stage || 'Not recorded'],
            ['Route', investigation.route || debug.plan?.route || 'Not recorded'],
            ['Model calls', investigation.model_calls ?? 'Not recorded'],
            ['Tool calls', investigation.tool_calls ?? 'Not recorded'],
            ['Evidence', `${evidence.length} records · ${evidence.filter(item => item.cited === true).length} cited`],
        ];
        const overview = `<dl class="debug-facts">${facts.map(([name, value]) =>
            `<div><dt>${escape(name)}</dt><dd>${escape(value)}</dd></div>`).join('')}</dl>`;
        const warnings = uniqueIssues.length ? `<div class="debug-warning"><strong>Recorded issues (${uniqueIssues.length})</strong><ul>${uniqueIssues.slice(0, 5).map(issue =>
            `<li>${escape(short(issue, 350))}</li>`).join('')}</ul>${uniqueIssues.length > 5 ? '<p>More issues in the full diagnostics below.</p>' : ''}</div>` : '';
        const timeline = steps.map((step, index) => {
            const args = step.args || {};
            const scope = [args.program || list(args.programs).join(', '), args.table,
                args.source_file, args.paragraph || args.identifier,
                args.start != null ? `lines ${args.start}–${args.end ?? args.start}` : '',
                args.terms ? `terms: ${list(args.terms).join(', ') || args.terms}` : ''].filter(Boolean).join(' · ');
            const label = step.tool || (step.review_response ? 'Answer review' : step.review || step.status || step.action ||
                (step.resolution_error ? 'Request resolution error' : step.resolved_request ? 'Request resolved' : 'Investigation step'));
            const state = step.status || (step.review_response ? (step.review_response.passed ? 'passed' : 'failed') : '');
            const meta = [state, step.returned != null ? `${step.returned} returned` : '',
                typeof step.duration_ms === 'number' ? `${(step.duration_ms / 1000).toFixed(2)}s` : ''].filter(Boolean).join(' · ');
            const detail = Object.fromEntries(Object.entries(step).filter(([key]) =>
                !['candidate_excerpt', 'candidate_answer', 'answer'].includes(key)));
            return fold(`${index + 1}. ${label}${meta ? ` · ${meta}` : ''}`, `${scope ? `<p class="debug-muted">${escape(scope)}</p>` : ''}<pre>${escape(short(JSON.stringify(detail, null, 2), 2500))}</pre>`);
        }).join('');
        const sources = [...groups].map(([label, items]) => fold(`${label} — ${items.length} records, ${items.filter(item => item.cited === true).length} cited`,
            `<table class="debug-table"><thead><tr><th>Reference</th><th>Use</th><th>Source preview</th></tr></thead><tbody>${items.map(item =>
                `<tr><td>${escape(item.source_id || item.evidence_id || `Record ${item.rank ?? '?'}`)}</td><td>${item.cited === true ? 'Cited' : 'Retrieved'}</td><td>${escape(evidencePreview(item))}</td></tr>`).join('')}</tbody></table><p class="debug-muted">Previews only. Complete recorded evidence is preserved in Full raw diagnostics.</p>`)).join('');
        const candidate = debug.candidate_answer ? fold('Unverified candidate — not an accepted answer',
            `<p class="debug-warning">Intermediate candidates are not verified answers.</p><pre>${escape(debug.candidate_answer)}</pre>`) : '';
        const attempts = [...list(debug.attempts), ...list(debug.subtasks)];
        const extras = attempts.length ? fold(`Attempts and claim checks (${attempts.length})`,
            `<pre>${escape(JSON.stringify(attempts, null, 2))}</pre>`) : '';
        // Complete metadata appears once, minified and escaped. No evidence is
        // removed; collapsed previews are for people, raw diagnostics for review.
        return `<details class="debug-panel${validation.passed === false ? ' rejected' : ''}"><summary><span>Debug details</span><small>${escape(status)} · ${evidence.length} evidence${traceId ? ` · trace ${escape(traceId.slice(0, 8))}` : ''}</small></summary><div class="debug-body">${overview}${warnings}${steps.length ? fold(`Execution timeline (${steps.length} steps)`, `<div class="debug-scroll">${timeline}</div>`) : ''}${evidence.length ? fold(`Evidence by file (${groups.size} groups · ${evidence.length} records)`, `<div class="debug-scroll">${sources}</div>`) : ''}${candidate}${extras}${fold('Full raw diagnostics — complete, unchanged values', `<p class="debug-muted">Intermediate candidates are not verified answers. Evidence may be excerpted by the API. Sensitive code may be present.</p><pre class="debug-raw">${escape(JSON.stringify(raw))}</pre>`)}</div></details>`;
    }
    function render({messages, sessionId, startedAt, exportedAt, timezone, includeDebug, pending = 0, running = false}) {
        const sections = messages.map((message, index) => {
            const sourceList = [...new Map((message.sources || []).map(source => [JSON.stringify([source.program,
                source.source_file || source.evidence_path || source.source_path || source.source_id]), source])).values()];
            const sources = sourceList.map(source => `<li>${escape(
                source.source_file || source.evidence_path || source.source_path || source.source_id || 'Source'
            )}${source.program ? ` — ${escape(source.program)}` : ''}</li>`).join('');
            const runtime = message.metadata?.runtime;
            const runtimeText = runtime ? `${runtimeSummary(runtime)}. Model: ${runtime.llm ?? 'unavailable'}; embedding: ${runtime.embedding ?? 'unavailable'}; collection: ${runtime.collection ?? 'unavailable'}; server completed: ${runtime.completed_at ?? 'unavailable'}` : '';
            // Debug-free exports never serialize the metadata object.
            const debug = includeDebug && message.metadata
                ? debugView(message.metadata.debug || {status: 'available'}, message.metadata.trace_id, message.metadata) : '';
            return `<section><h2>${index + 1}. ${message.role === 'user' ? 'You' : 'AI'}</h2><p class="meta">${escape(message.timestamp)}</p><div class="answer">${escape(message.content)}</div>${sources ? fold(`Sources (${sourceList.length} unique)`, `<ul>${sources}</ul>`) : ''}${runtimeText ? `<p class="meta">${escape(runtimeText)}</p>` : ''}${debug}</section>`;
        }).join('\n');
        return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>COBOL RAG chat — ${escape(exportedAt)}</title><style>body{max-width:960px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui,sans-serif;color:#182230}h1,h2,h3{line-height:1.3}section{border-top:1px solid #ccd3db;padding:20px 0}.meta{color:#526070;font-size:13px;overflow-wrap:anywhere}.answer,pre{white-space:pre-wrap;overflow-wrap:anywhere}pre{background:#f2f4f7;padding:16px;font-size:12px}summary{cursor:pointer;font-weight:600;overflow-wrap:anywhere}details{border:1px solid #d5dbe3;border-radius:6px;margin:8px 0}summary{padding:8px 10px}.debug-body{padding:10px}.debug-facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:8px;margin:0}.debug-facts div{display:flex;gap:8px;font-size:13px}.debug-facts dt{color:#526070}.debug-facts dd{margin:0}.debug-scroll{max-height:340px;overflow:auto;padding:0 8px 8px}.debug-body pre{max-height:260px;overflow:auto;margin:8px;padding:10px;font-size:12px}.debug-body .debug-raw{max-height:340px}.debug-warning{color:#9e2929;max-height:180px;overflow:auto}.debug-muted{font-size:12px;color:#526070;margin:8px}.debug-panel>summary{display:flex;flex-wrap:wrap;justify-content:space-between;gap:8px}.debug-panel small{color:#526070}.rejected{border-color:#d68e8e}.debug-table{width:100%;border-collapse:collapse;font-size:12px;table-layout:fixed}.debug-table td,.debug-table th{text-align:left;vertical-align:top;padding:6px;border-bottom:1px solid #d5dbe3;overflow-wrap:anywhere}.debug-table th:first-child{width:20%}.debug-table th:nth-child(2){width:14%}header{background:#f2f4f7;padding:20px}@media print{body{margin:0}details{display:block}}</style></head><body><header><h1>COBOL RAG — Chat transcript</h1><p>Exported: ${escape(exportedAt)} (UTC)<br>Local timezone: ${escape(timezone)}<br>Session: ${escape(sessionId)}<br>Browser transcript started: ${escape(startedAt)}<br>Messages: ${messages.length} · Debug details: ${includeDebug ? 'included' : 'excluded'}<br>Export format version: 2</p><p>This snapshot contains messages recorded in this browser tab since opening or clearing it, not earlier server history. ${running ? 'An answer was still running at export time. ' : ''}${pending ? `${pending} queued questions are not included. ` : ''}Unavailable runtime metadata is not inferred. Sources and debug information may contain private code; review before sharing.</p></header>${sections}</body></html>`;
    }
    root.ChatExport = {render, runtimeSummary, debugView};
    if (typeof module !== 'undefined') module.exports = {render, runtimeSummary, debugView};
})(typeof globalThis !== 'undefined' ? globalThis : window);
