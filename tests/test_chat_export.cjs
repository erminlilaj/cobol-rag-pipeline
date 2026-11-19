const {test} = require('node:test');
const assert = require('node:assert/strict');
const {render, runtimeSummary, debugView} = require('../ui/chat-export.js');

const snapshot = {
    sessionId: 'session-123', startedAt: '2026-09-24T10:00:00Z',
    exportedAt: '2026-09-24T10:01:00Z', timezone: 'Europe/Rome',
    messages: [{role: 'user', content: 'Cosa fa A? <script>alert(1)</script>', timestamp: '10:00'},
        {role: 'assistant', content: 'A processes records.\nSecond line.', timestamp: '10:01',
            sources: [{source_file: 'A.CBL', program: 'A'}],
            metadata: {trace_id: 'trace-123', debug: {candidate_answer: 'PRIVATE-CANDIDATE'},
                runtime: {llm: 'model-A', embedding: 'embed-A', collection: 'corpus',
                    completed_at: '2026-09-24T10:01:00Z', duration_ms: 123}}}],
};
test('clean export contains messages, provenance and dates but no hidden debug', () => {
    const html = render({...snapshot, includeDebug: false});
    for (const text of ['A.CBL', 'model-A', 'session-123', snapshot.exportedAt, 'Second line.']) {
        assert.ok(html.includes(text));
    }
    assert.ok(!html.includes('PRIVATE-CANDIDATE'));
    assert.ok(!html.includes('trace-123'));
    assert.ok(!html.includes('<script>'));
    assert.ok(html.includes('&lt;script&gt;'));
});
test('debug export preserves raw diagnostics, safely escaped', () => {
    const html = render({...snapshot, includeDebug: true});
    assert.ok(html.includes('PRIVATE-CANDIDATE'));
    assert.ok(html.includes('trace-123'));
    assert.ok(html.includes('Intermediate candidates are not verified answers'));
});
test('partial snapshots and missing metadata are explicit', () => {
    const html = render({...snapshot, messages: [snapshot.messages[0]], running: true, pending: 2});
    assert.ok(html.includes('still running'));
    assert.ok(html.includes('2 queued questions'));
    assert.ok(!html.includes('Model: undefined'));
});

test('answer metrics show server time and provider input/output totals', () => {
    const runtime = {duration_ms: 65123, token_usage: {input_tokens: 12000,
        output_tokens: 250, total_tokens: 12250, complete: true, reported_calls: 3, model_calls: 3}};
    assert.equal(runtimeSummary(runtime), 'Time: 1m 5.1s · Tokens: 12,250 (in 12,000 / out 250)');
    const messages = [{role: 'assistant', content: 'Answer', metadata: {runtime}}];
    assert.ok(render({...snapshot, messages, includeDebug: false}).includes('Tokens: 12,250'));
});

test('missing and partial token reports never look like complete zero usage', () => {
    assert.equal(runtimeSummary({duration_ms: 1250}), 'Time: 1.3s · Tokens: unavailable');
    assert.match(runtimeSummary({duration_ms: NaN}), /Time: unavailable/);
    assert.match(runtimeSummary({token_usage: {input_tokens: 10, output_tokens: 2,
        total_tokens: 12, reported_calls: 1, model_calls: 3, complete: false}}),
        /reported subtotal.*12.*1 \/ 3 calls reported/);
    assert.match(runtimeSummary({token_usage: {input_tokens: 0, output_tokens: 0,
        total_tokens: 0, reported_calls: 1, model_calls: 1, complete: true}}), /Tokens: 0/);
    assert.match(runtimeSummary({token_usage: {input_tokens: '<script>', output_tokens: 0,
        total_tokens: 0, complete: true}}), /Tokens: unavailable/);
});

function rawDiagnostics(html) {
    const raw = html.match(/<pre class="debug-raw">([\s\S]*?)<\/pre>/)[1];
    const entities = {'&lt;': '<', '&gt;': '>', '&quot;': '"', '&#39;': "'", '&amp;': '&'};
    return JSON.parse(raw.replace(/&(?:lt|gt|quot|#39|amp);/g, entity => entities[entity]));
}

const detailed = {
    status: 'rejected', validation: {passed: false, stage: 'review', reasons: ['Missing source body']},
    investigation: {model_calls: 4, tool_calls: 3, steps: [
        {tool: 'source_outline', args: {program: 'A', terms: ['ITEM']}, status: 'ok', returned: 2},
        {tool: 'source_range', args: {program: 'A', source_file: 'A.CBL', start: 20, end: 40},
            status: 'ok', returned: 21, duration_ms: 125},
        {status: 'error', error: 'Invalid depth <script>'},
    ]},
    retrieval: {evidence: Array.from({length: 250}, (_, i) => ({source_id: `E${i + 1}`,
        source_file: i < 200 ? 'A.CBL' : 'B.CBL', program: i < 200 ? 'A' : 'B', cited: i === 0,
        excerpt: JSON.stringify({text: 'MOVE AMOUNT TO RESULT '.repeat(100), line: i + 1})}))},
    unknown_future_field: {exact: 'Retain \n whitespace <script> & "quotes"'},
};

test('compact debug folds timeline and file groups, never opens hundreds of records', () => {
    const html = debugView(detailed, 'trace-abcd');
    assert.ok(html.includes('Execution timeline (3 steps)'));
    assert.ok(html.includes('Evidence by file (2 groups · 250 records)'));
    assert.ok(html.includes('A · A.CBL — 200 records, 1 cited'));
    assert.ok(html.includes('2. source_range · ok · 21 returned · 0.13s'));
    assert.ok(html.includes('lines 20–40'));
    assert.doesNotMatch(html, /<details[^>]*\sopen(?:\s|>)/);
    assert.ok(html.includes('Invalid depth &lt;script&gt;'));
    assert.deepEqual(rawDiagnostics(html), detailed);
});

test('full export preserves complete original metadata once with bounded visual panels', () => {
    const metadata = {debug: detailed, runtime: {llm: 'Qwen', duration_ms: 1500}, trace_id: 'trace-abcd'};
    const messages = [{role: 'assistant', content: 'A partial answer.', metadata}];
    const html = render({...snapshot, messages, includeDebug: true});
    assert.deepEqual(rawDiagnostics(html), metadata);
    assert.equal((html.match(/class="debug-raw"/g) || []).length, 1);
    assert.ok(html.includes('max-height:340px;overflow:auto'));
    assert.ok(html.includes('Export format version: 2'));
    assert.ok(!html.includes('<script>'));
    assert.ok(!html.includes('undefined'));
    assert.deepEqual(metadata.debug, detailed);
});

test('source list deduplicates repeated provenance without mixing different programs', () => {
    const source = {source_file: 'SHARED.CPY', program: 'A'};
    const messages = [{role: 'assistant', content: 'Answer', sources: [source, source,
        {source_file: 'SHARED.CPY', program: 'B'}]}];
    const html = render({...snapshot, messages});
    assert.ok(html.includes('Sources (2 unique)'));
    assert.equal((html.match(/<li>/g) || []).length, 2);
});

test('unknown validation is not presented as a successful validation', () => {
    const html = debugView({status: 'available'});
    assert.ok(html.includes('<dt>Validation</dt><dd>Not recorded</dd>'));
    assert.ok(!html.includes('<dd>Passed</dd>'));
    assert.equal(debugView(null), '');
    assert.equal(debugView({}), '');
});

test('legacy plan traces and repeated failures remain readable without losing raw events', () => {
    const debug = {validation: {passed: false, reasons: ['Same issue', 'Same issue']},
        plan: {steps: [{resolution_error: 'Same issue'}, {review_response: {passed: false,
            issues: ['Scope mismatch']}}]}};
    const html = debugView(debug);
    assert.ok(html.includes('Execution timeline (2 steps)'));
    assert.ok(html.includes('Recorded issues (2)'));
    assert.deepEqual(rawDiagnostics(html), debug);
});

test('clean exports never include compact hidden evidence or candidates', () => {
    const html = render({...snapshot, messages: [{role: 'assistant', content: 'Answer',
        metadata: {debug: detailed}}], includeDebug: false});
    for (const secret of ['unknown_future_field', 'Invalid depth', 'E250', 'debug-raw\">']) {
        assert.ok(!html.includes(secret));
    }
});
