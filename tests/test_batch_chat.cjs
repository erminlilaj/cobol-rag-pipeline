const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const ui = path.join(__dirname, '../ui');
const script = fs.readFileSync(path.join(ui, 'app.js'), 'utf8');
const html = fs.readFileSync(path.join(ui, 'index.html'), 'utf8');
const tick = () => new Promise(resolve => setImmediate(resolve));

function page() {
    function element() {
        const listeners = new Map();
        return {
            children: [],
            value: '', textContent: '', hidden: false, open: false, style: {},
            addEventListener(type, handler) { listeners.set(type, handler); },
            emit(type) { listeners.get(type)({preventDefault() {}}); },
            showModal() { this.open = true; },
            close() { this.open = false; },
            appendChild(child) { this.children.push(child); },
            append(...children) { this.children.push(...children); },
            prepend(child) { this.children.unshift(child); },
            scrollTo() {}, focus() {},
        };
    }
    const elements = new Map(Array.from(html.matchAll(/\bid="([^"]+)"/g),
        match => [match[1], element()]));
    const requests = [];
    const messages = [];
    const context = vm.createContext({
        AbortController, console, ChatExport: require('../ui/chat-export.js'),
        document: {getElementById: id => elements.get(id), addEventListener() {}, createElement: element},
        window: {localStorage: {getItem: () => 'test-session'}, addEventListener() {}},
        messages,
        request(url, options) {
            if (url !== '/api/chat') return Promise.resolve({});
            return new Promise((resolve, reject) => {
                requests.push({message: JSON.parse(options.body).message, resolve});
                options.signal.addEventListener('abort', () => {
                    reject(Object.assign(new Error('aborted'), {name: 'AbortError'}));
                }, {once: true});
            });
        },
    });
    vm.runInContext(script + `
        apiFetch = request;
        const originalAppendMessage = appendMessage;
        createWaitingExperience = () => ({element: {remove() {}}, destroy() {}});
        appendMessage = (role, text) => messages.push([role, text]);
    `, context);
    return {get: id => elements.get(id), requests, messages,
            stop: () => vm.runInContext('stopChat()', context),
            send: () => vm.runInContext('sendMessage()', context),
            reset: () => vm.runInContext('resetChat()', context),
            session: () => vm.runInContext('chatSessionId', context),
            observe: value => { context.incoming = value; vm.runInContext('observeConversationContext(incoming)', context); },
            showAnswer: metadata => { context.metadata = metadata; vm.runInContext("originalAppendMessage('assistant', 'Supported answer', [], metadata)", context); }};
}

test('answer bubble displays metrics outside collapsed debug details', () => {
    const chat = page();
    chat.showAnswer({runtime: {duration_ms: 2500, token_usage: {input_tokens: 1000,
        output_tokens: 200, total_tokens: 1200, complete: true, reported_calls: 2, model_calls: 2}}});
    const message = chat.get('chat-history').children.at(-1);
    const content = message.children.find(child => child.className === 'content');
    const metrics = content.children.find(child => child.className === 'answer-runtime');
    assert.equal(metrics.textContent, 'Time: 2.5s · Tokens: 1,200 (in 1,000 / out 200)');
    assert.match(metrics.title, /planning.*review.*retries/);
});

test('new tabs and empty refreshed transcripts get different conversation IDs', () => {
    const first = page(), second = page();
    assert.notEqual(first.session(), second.session());
    assert.notEqual(first.session(), 'test-session');
});

test('reset rotates chat identity as well as clearing the visible transcript', async () => {
    const chat = page(), oldId = chat.session();
    await chat.reset();
    assert.notEqual(chat.session(), oldId);
    assert.match(chat.messages.at(-1)[1], /memory cleared/);
});

test('server restart is disclosed only for a memory-enabled conversation', () => {
    const chat = page();
    chat.observe({id: 'old-server', memory_enabled: true});
    chat.observe({id: 'new-server', memory_enabled: true});
    assert.equal(chat.messages.length, 1);
    assert.match(chat.messages[0][1], /processed without the previous chat context/);
    chat.observe({id: 'another-server', memory_enabled: false});
    assert.equal(chat.messages.length, 1);
});

test('a cancelled server response is discarded rather than shown as an answer', async () => {
    const chat = page();
    chat.get('chat-input').value = 'A question';
    const sending = chat.send();
    chat.requests[0].resolve({answer: 'Obsolete answer', execution_mode: 'cancelled'});
    await sending;
    assert.ok(chat.messages.some(message => /discarded/.test(message[1])));
    assert.ok(!chat.messages.some(message => message[1] === 'Obsolete answer'));
});

test('batch dialog counts nonblank lines and preserves an unsubmitted draft', () => {
    const {get, requests} = page();
    get('open-batch-btn').emit('click');
    assert.equal(get('batch-dialog').open, true);
    assert.equal(get('batch-run-btn').disabled, true);
    get('batch-input').value = '  First?  \r\n\r\nSecond?\n ';
    get('batch-input').emit('input');
    assert.equal(get('batch-count').textContent, '2 questions');
    assert.equal(get('batch-run-btn').disabled, false);
    get('batch-cancel-btn').emit('click');
    assert.equal(get('batch-dialog').open, false);
    assert.equal(requests.length, 0);
    get('open-batch-btn').emit('click');
    assert.equal(get('batch-count').textContent, '2 questions');
});

test('50 pasted questions appear as ordinary chat turns in order, one request at a time', async () => {
    const {get, requests, messages} = page();
    const questions = Array.from({length: 50}, (_, i) => `Question ${i + 1}?`);
    get('chat-input').value = 'Keep my normal draft';
    get('open-batch-btn').emit('click');
    get('batch-input').value = questions.join('\r\n\r\n');
    get('batch-input').emit('input');
    get('batch-form').emit('submit');
    assert.equal(get('batch-dialog').open, false);
    assert.equal(get('batch-input').value, '');
    assert.equal(get('chat-input').value, 'Keep my normal draft');
    for (const [i, question] of questions.entries()) {
        assert.equal(requests.length, i + 1);
        assert.equal(requests[i].message, question);
        assert.deepEqual([...messages[i * 2]], ['user', question]);
        requests[i].resolve({answer: `Answer ${i + 1}`});
        await tick();
        assert.deepEqual([...messages[i * 2 + 1]], ['assistant', `Answer ${i + 1}`]);
    }
    assert.equal(get('queue-bar').hidden, true);
    assert.equal(get('stop-btn').hidden, true);
});

test('batch added during a normal chat shares its queue; Stop discards remaining questions', async () => {
    const {get, requests, messages, send, stop} = page();
    get('chat-input').value = 'Current question';
    const running = send();
    get('open-batch-btn').emit('click');
    get('batch-input').value = 'Queued one\nQueued two';
    get('batch-form').emit('submit');
    assert.equal(requests.length, 1);
    assert.equal(get('queue-status').textContent, '2 questions queued');
    requests[0].resolve({answer: 'First answer'});
    await tick();
    assert.equal(requests[1].message, 'Queued one');
    await stop();
    await running;
    assert.equal(requests.length, 2);
    assert.equal(get('queue-bar').hidden, true);
    assert.ok(messages.some(([, text]) => text === '1 queued question cancelled.'));
});
