const {test} = require('node:test');
const assert = require('node:assert/strict');
const {render, runtimeSummary} = require('../ui/chat-export.js');

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
