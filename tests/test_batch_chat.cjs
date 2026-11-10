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
            value: '', textContent: '', hidden: false, open: false, style: {},
            addEventListener(type, handler) { listeners.set(type, handler); },
            emit(type) { listeners.get(type)({preventDefault() {}}); },
            showModal() { this.open = true; },
            close() { this.open = false; },
            appendChild() {}, scrollTo() {}, focus() {},
        };
    }
    const elements = new Map(Array.from(html.matchAll(/\bid="([^"]+)"/g),
        match => [match[1], element()]));
    const requests = [];
    const messages = [];
    const context = vm.createContext({
        AbortController, console,
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
        createWaitingExperience = () => ({element: {remove() {}}, destroy() {}});
        appendMessage = (role, text) => messages.push([role, text]);
    `, context);
    return {get: id => elements.get(id), requests, messages,
            stop: () => vm.runInContext('stopChat()', context),
            send: () => vm.runInContext('sendMessage()', context)};
}

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
