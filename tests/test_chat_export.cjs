const {test} = require('node:test');
const assert = require('node:assert/strict');
const {render} = require('../ui/chat-export.js');

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
