const modal = document.getElementById('modal');
const modalTitle = document.getElementById('modal-title');
const modalBody = document.getElementById('modal-body');
const chatHistory = document.getElementById('chat-history');
const chatInput = document.getElementById('chat-input');
const sendButton = document.getElementById('send-btn');
const stopButton = document.getElementById('stop-btn');
const queueBar = document.getElementById('queue-bar');
const queueStatus = document.getElementById('queue-status');
const queueList = document.getElementById('queue-list');
const batchDialog = document.getElementById('batch-dialog');
const batchInput = document.getElementById('batch-input');
const batchCount = document.getElementById('batch-count');
const batchRunButton = document.getElementById('batch-run-btn');
const statusIndicator = document.getElementById('status-indicator');
const collectionLabel = document.getElementById('collection-label');
const treeFilter = document.getElementById('tree-filter');

let selectedInboxPaths = new Set();
let inboxTreeData = null;
let transcript = [];
let transcriptStartedAt = new Date().toISOString();

function downloadChat(includeDebug) {
    if (!transcript.length) {
        showModal('Download Chat', '<p>Send a message first. The initial welcome message is not a conversation.</p>');
        return;
    }
    const exportedAt = new Date().toISOString();
    const html = ChatExport.render({messages: transcript, sessionId: chatSessionId,
        startedAt: transcriptStartedAt, exportedAt,
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
        includeDebug, pending: pendingQuestions.length, running: Boolean(runningRequest)});
    const url = URL.createObjectURL(new Blob([html], {type: 'text/html;charset=utf-8'}));
    const link = document.createElement('a');
    link.href = url;
    link.download = `cobol-rag-chat-${exportedAt.replace(/[:.]/g, '-')}-${includeDebug ? 'debug' : 'clean'}.html`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function newConversationId() {
    return window.crypto?.randomUUID?.()
        || `chat-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}
// Each visible transcript owns its conversation. No localStorage or duplicated
// sessionStorage ID: new tabs and refreshed empty transcripts cannot inherit
// hidden context from another chat. Server memory remains opt-in.
let chatSessionId = newConversationId();
let serverConversationContext = null;

function observeConversationContext(context) {
    if (!context?.id) return;
    if (context.memory_enabled && serverConversationContext && serverConversationContext.id !== context.id) {
        appendMessage('assistant', 'The server conversation context was reset or expired. This question was processed without the previous chat context; restate the program and entity for follow-up questions.');
    }
    serverConversationContext = context;
}

function scrollChatToBottom() {
    chatHistory.scrollTo({
        top: chatHistory.scrollHeight,
        behavior: 'smooth',
    });
}

function toggleAccordion(element) {
    element.parentElement.classList.toggle('active');
}

function handleEnter(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

function showModal(title, contentHTML) {
    modalTitle.textContent = title;
    modalBody.innerHTML = contentHTML;
    modal.style.display = 'flex';
}

function closeModal() {
    modal.style.display = 'none';
}

window.addEventListener('click', event => {
    if (event.target === modal) {
        closeModal();
    }
});

function escapeHTML(value) {
    return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
}

function renderMarkdownLite(text) {
    const escaped = escapeHTML(text);
    const withCodeBlocks = escaped.replace(/```([\s\S]*?)```/g, (_match, code) => `<pre><code>${code.trim()}</code></pre>`);
    const paragraphs = withCodeBlocks
        .split(/\n{2,}/)
        .map(block => {
            if (block.startsWith('<pre>')) return block;
            const lines = block.split('\n');
            if (lines.every(line => line.trim().startsWith('- '))) {
                return `<ul>${lines.map(line => `<li>${line.trim().slice(2)}</li>`).join('')}</ul>`;
            }
            return `<p>${lines.join('<br>')}</p>`;
        });
    return paragraphs.join('');
}

function renderDebugDetails(debug, traceId = '') {
    if (!debug || !Object.keys(debug).length) return '';

    const validation = debug.validation || {};
    const retrieval = debug.retrieval || {};
    const attempts = Array.isArray(debug.attempts) ? debug.attempts : [];
    const subtasks = Array.isArray(debug.subtasks) ? debug.subtasks : [];
    const evidence = Array.isArray(retrieval.evidence) ? retrieval.evidence : [];
    const reasons = Array.isArray(validation.reasons) ? validation.reasons : [];
    const disposition = debug.evidence_disposition || {};
    const status = debug.status || 'available';
    const rejected = status === 'rejected' || validation.passed === false;

    const candidate = debug.candidate_answer ? `
        <section class="debug-section debug-warning-section">
            <h5>Unverified candidate answer</h5>
            <p class="debug-warning">This is an intermediate result for debugging. It was not accepted as the trusted answer.</p>
            <pre>${escapeHTML(debug.candidate_answer)}</pre>
        </section>` : '';

    const validationBlock = `
        <section class="debug-section">
            <h5>Validation</h5>
            <dl class="debug-facts">
                <div><dt>Stage</dt><dd>${escapeHTML(validation.stage || 'none')}</dd></div>
                <div><dt>Passed</dt><dd>${validation.passed === false ? 'No' : 'Yes'}</dd></div>
                <div><dt>Guard</dt><dd>${escapeHTML(debug.guard_status || 'not applicable')}</dd></div>
                <div><dt>Execution</dt><dd>${escapeHTML((debug.execution_strategy || 'unknown').replaceAll('_', ' '))}</dd></div>
                <div><dt>Evidence state</dt><dd>${escapeHTML((disposition.state || 'unknown').replaceAll('_', ' '))}</dd></div>
            </dl>
            ${reasons.length
                ? `<ul class="debug-reasons">${reasons.map(reason => `<li>${escapeHTML(reason)}</li>`).join('')}</ul>`
                : '<p class="debug-muted">No validation failure reason was recorded.</p>'}
        </section>`;

    const attemptBlock = attempts.length ? `
        <section class="debug-section">
            <h5>Answer attempts (${attempts.length})</h5>
            ${attempts.map((attempt, index) => {
                const attemptReasons = Array.isArray(attempt.reasons) ? attempt.reasons : [];
                const attemptAnswer = attempt.rendered_answer || attempt.candidate_answer || '';
                return `<details class="debug-nested">
                    <summary>Attempt ${index + 1}: ${escapeHTML(attempt.stage || 'unknown')} · ${attempt.passed ? 'accepted' : 'rejected'}</summary>
                    ${attemptReasons.length ? `<p><strong>Reasons:</strong> ${escapeHTML(attemptReasons.join(', '))}</p>` : ''}
                    ${attemptAnswer ? `<pre>${escapeHTML(attemptAnswer)}</pre>` : ''}
                </details>`;
            }).join('')}
        </section>` : '';

    const subtaskBlock = subtasks.length ? `
        <section class="debug-section">
            <h5>Semantic claim plan (${subtasks.length})</h5>
            ${subtasks.map((subtask, index) => {
                const subtaskAttempts = Array.isArray(subtask.attempts) ? subtask.attempts : [];
                const subtaskSources = Array.isArray(subtask.sources) ? subtask.sources : [];
                const subtaskReasons = Array.isArray(subtask.reasons) ? subtask.reasons : [];
                return `<details class="debug-nested">
                    <summary>Claim ${index + 1}: ${escapeHTML(subtask.capability || 'unknown')} · ${subtask.passed ? 'verified' : 'unresolved'}</summary>
                    <p>${escapeHTML(subtask.description || '')}</p>
                    <p class="debug-muted"><strong>Entities:</strong> ${escapeHTML((subtask.entity_values || []).join(', ') || 'program-wide')} · <strong>Sources:</strong> ${subtaskSources.length} · <strong>Attempts:</strong> ${subtaskAttempts.length}</p>
                    ${subtaskReasons.length ? `<p><strong>Reasons:</strong> ${escapeHTML(subtaskReasons.join(', '))}</p>` : ''}
                    ${subtaskAttempts.map(attempt => `<details class="debug-nested">
                        <summary>${escapeHTML(attempt.stage || 'attempt')} · ${attempt.passed ? 'accepted' : 'rejected'}</summary>
                        ${(attempt.reasons || []).length ? `<p><strong>Reasons:</strong> ${escapeHTML(attempt.reasons.join(', '))}</p>` : ''}
                        ${attempt.candidate_answer ? `<pre>${escapeHTML(attempt.candidate_answer)}</pre>` : ''}
                    </details>`).join('')}
                </details>`;
            }).join('')}
        </section>` : '';

    const evidenceBlock = `
        <section class="debug-section">
            <h5>${debug.investigation ? 'Evidence retrieved' : 'Evidence inspected'} (${evidence.length})</h5>
            ${evidence.length ? evidence.map(item => {
                const label = item.source_file || item.source_id || `Evidence ${item.rank || ''}`;
                const descriptors = [item.chunk_type, item.program, item.entity_key].filter(Boolean).join(' · ');
                const score = item.score === null || item.score === undefined ? '' : ` · score ${Number(item.score).toFixed(4)}`;
                return `<details class="debug-nested">
                    <summary>${escapeHTML(label)}${score}</summary>
                    ${descriptors ? `<p class="debug-muted">${escapeHTML(descriptors)}</p>` : ''}
                    ${debug.investigation ? `<p class="debug-muted">${item.cited ? 'Cited in answer' : 'Retrieved, not cited'}${item.excerpt_truncated ? ' · excerpt truncated' : ''}</p>` : ''}
                    <pre>${escapeHTML(item.excerpt || 'No text excerpt available.')}</pre>
                </details>`;
            }).join('') : '<p class="debug-muted">No evidence record was attached to this route.</p>'}
        </section>`;

    const planBlock = `
        <section class="debug-section">
            <h5>${debug.investigation ? 'Investigation request and execution trace' : 'Query plan'}</h5>
            <pre>${escapeHTML(JSON.stringify(debug.plan || {}, null, 2))}</pre>
        </section>`;

    return `<details class="debug-panel${rejected ? ' rejected' : ''}">
        <summary><span>Debug details</span><small>${escapeHTML(status)}${traceId ? ` · trace ${escapeHTML(traceId.slice(0, 8))}` : ''}</small></summary>
        <div class="debug-body">
            ${candidate}
            ${validationBlock}
            ${subtaskBlock}
            ${attemptBlock}
            ${evidenceBlock}
            ${planBlock}
        </div>
    </details>`;
}

function setStatus(state, text) {
    statusIndicator.className = `status ${state}`;
    statusIndicator.textContent = text;
}

async function apiFetch(url, options) {
    const requestOptions = options || {};
    const headers = new Headers(requestOptions.headers || {});
    headers.set('X-Session-ID', chatSessionId);
    const response = await fetch(url, { ...requestOptions, headers });
    const contentType = response.headers.get('content-type') || '';
    const payload = contentType.includes('application/json') ? await response.json() : await response.text();
    if (!response.ok) {
        const detail = typeof payload === 'object' ? payload.detail || JSON.stringify(payload) : payload;
        throw new Error(detail || `Request failed with ${response.status}`);
    }
    return payload;
}

function renderTree(node) {
    const wrapper = document.createElement('div');
    wrapper.className = 'tree-node';
    wrapper.dataset.name = `${node.name} ${node.path}`.toLowerCase();

    const content = document.createElement('label');
    content.className = 'tree-node-content';
    content.title = node.path;

    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.value = node.path;
    checkbox.checked = selectedInboxPaths.has(node.path);
    checkbox.addEventListener('change', event => {
        handleCheckboxChange(event.target.checked, node, wrapper);
    });

    const icon = document.createElement('span');
    icon.textContent = node.is_dir ? '▣' : '◻';

    const label = document.createElement('span');
    label.className = 'tree-label';
    label.textContent = node.name;

    content.append(checkbox, icon, label);
    wrapper.appendChild(content);

    if (node.is_dir && node.children?.length) {
        const children = document.createElement('div');
        children.className = 'tree-children';
        node.children.forEach(child => children.appendChild(renderTree(child)));
        wrapper.appendChild(children);
    }

    return wrapper;
}

function handleCheckboxChange(checked, node, element) {
    if (checked) {
        selectedInboxPaths.add(node.path);
    } else {
        selectedInboxPaths.delete(node.path);
    }

    if (!node.is_dir) return;
    element.querySelectorAll('.tree-children input[type="checkbox"]').forEach(checkbox => {
        checkbox.checked = checked;
        if (checked) {
            selectedInboxPaths.add(checkbox.value);
        } else {
            selectedInboxPaths.delete(checkbox.value);
        }
    });
}

function drawInboxTree() {
    const container = document.getElementById('inbox-tree');
    container.innerHTML = '';
    if (!inboxTreeData) {
        container.innerHTML = '<div class="empty-state">No inbox data loaded.</div>';
        return;
    }
    container.appendChild(renderTree(inboxTreeData));
    filterTree();
}

function filterTree() {
    const needle = treeFilter.value.trim().toLowerCase();
    document.querySelectorAll('.tree-node').forEach(node => {
        node.classList.toggle('hidden', Boolean(needle) && !node.dataset.name.includes(needle));
    });
}

async function loadHealth() {
    try {
        const data = await apiFetch('/api/health');
        setStatus('online', 'Online');
        collectionLabel.textContent = `${data.collection} · ${data.embedding}`;
    } catch (error) {
        setStatus('offline', 'Offline');
        collectionLabel.textContent = 'API unavailable';
    }
}

async function loadInbox() {
    const container = document.getElementById('inbox-tree');
    container.innerHTML = '<div class="empty-state">Loading inbox...</div>';
    try {
        inboxTreeData = await apiFetch('/api/inbox');
        selectedInboxPaths.clear();
        drawInboxTree();
    } catch (error) {
        container.innerHTML = `<div class="empty-state">Could not load inbox: ${escapeHTML(error.message)}</div>`;
    }
}

async function syncSelected() {
    const button = document.getElementById('btn-sync');
    button.textContent = 'Syncing...';
    button.disabled = true;

    try {
        const payload = { paths: selectedInboxPaths.size ? Array.from(selectedInboxPaths) : null };
        const data = await apiFetch('/api/sync', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        showModal('Sync Complete', `
            <p><strong>Collection:</strong> ${escapeHTML(data.collection)}</p>
            <p><strong>Processed:</strong> ${escapeHTML(data.documents_processed)}</p>
            <p><strong>Added:</strong> ${escapeHTML(data.added)}</p>
            <p><strong>Updated:</strong> ${escapeHTML(data.updated)}</p>
            <p><strong>Skipped:</strong> ${escapeHTML(data.skipped)}</p>
        `);
        await loadHealth();
    } catch (error) {
        showModal('Sync Failed', `<p>${escapeHTML(error.message)}</p>`);
    } finally {
        button.textContent = 'Sync Selected';
        button.disabled = false;
    }
}

async function fetchConfig() {
    try {
        const data = await apiFetch('/api/config');
        showModal('Configuration', `<pre>${escapeHTML(JSON.stringify(data, null, 2))}</pre>`);
    } catch (error) {
        showModal('Config Error', `<p>${escapeHTML(error.message)}</p>`);
    }
}

async function fetchIndexInfo() {
    try {
        const data = await apiFetch('/api/index-info');
        showModal('Index Info', `<pre>${escapeHTML(JSON.stringify(data, null, 2))}</pre>`);
    } catch (error) {
        showModal('Index Error', `<p>${escapeHTML(error.message)}</p>`);
    }
}

async function resetCollection() {
    if (!confirm('Reset the configured Chroma collection?')) return;
    try {
        const data = await apiFetch('/api/reset', { method: 'POST' });
        showModal('Reset Collection', `<p>${escapeHTML(data.message)}</p>`);
        await loadHealth();
    } catch (error) {
        showModal('Reset Failed', `<p>${escapeHTML(error.message)}</p>`);
    }
}

function activityPhase(phase) {
    return ({resolution: 'Resolving the request and scope', decision: 'Selecting evidence tools or preparing the answer',
        review: 'Checking the answer against evidence', partial_synthesis: 'Preparing a supported partial answer',
        input_translation: 'Translating the question', output_translation: 'Translating the answer',
        output_translation_repair: 'Repairing the answer translation'})[phase] || String(phase).replaceAll('_', ' ');
}

function activityRow(event) {
    const row = document.createElement('li');
    row.className = 'activity-event activity-' + String(event.kind || 'action').replace(/[^a-z_]/g, '');
    const d = event.details || {};
    row.textContent = '[' + ((event.elapsed_ms || 0) / 1000).toFixed(1) + 's] ' + event.message
        + (d.phase ? ' · ' + activityPhase(d.phase) : '')
        + (d.tool ? ' · ' + d.tool : '')
        + (d.path ? '\n' + d.path : '')
        + (d.source_file ? '\n' + d.source_file + (d.first_line != null ? ':' + d.first_line + '–' + d.last_line : '') : '')
        + (d.paragraph ? ' · paragraph ' + d.paragraph : '')
        + (d.scope ? '\n' + JSON.stringify(d.scope) : '')
        + (d.returned != null ? ' · ' + d.returned + ' rows' : '')
        + (d.checks?.length ? '\nValidation findings: ' + JSON.stringify(d.checks) : '')
        + (d.duration_ms != null ? ' · ' + (d.duration_ms / 1000).toFixed(2) + 's' : '')
        + (d.token_usage ? ' · in ' + d.token_usage.input_tokens + ' / out ' + d.token_usage.output_tokens + ' tokens' : '');
    if (Object.keys(d).length) {
        const details = document.createElement('details');
        const summary = document.createElement('summary');
        summary.textContent = 'Inspect parameters, file details and evidence';
        const raw = document.createElement('pre');
        raw.textContent = JSON.stringify(d, null, 2);
        details.append(summary, raw);
        row.appendChild(details);
    }
    if (Array.isArray(d.source_preview) && d.source_preview.length) {
        const source = document.createElement('pre');
        source.className = 'activity-source-preview';
        source.textContent = 'Recorded source excerpt (not a verified answer):\n' + d.source_preview.map(r =>
            (r.source_file || '') + ':' + (r.line ?? r.line_start ?? '?') + '  ' + (r.text || '')).join('\n');
        row.appendChild(source);
    }
    return row;
}

function appendMessage(role, content, sources = [], metadata = null) {
    transcript.push({role, content, sources, metadata, timestamp: new Date().toISOString()});
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = role === 'user' ? 'U' : 'AI';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'content';
    contentDiv.innerHTML = renderMarkdownLite(content);

    if (role === 'assistant' && metadata?.runtime) {
        const metrics = document.createElement('div');
        metrics.className = 'answer-runtime';
        metrics.textContent = ChatExport.runtimeSummary(metadata.runtime);
        metrics.title = 'Server processing time (not queue wait). Provider-reported input/output tokens across planning, answers, review and retries; excludes embeddings. Missing counts are not estimated.';
        contentDiv.appendChild(metrics);
    }

    if (role === 'assistant' && metadata?.execution_mode) {
        const metaDiv = document.createElement('div');
        metaDiv.className = 'answer-meta';
        const category = metadata.plan?.category || metadata.route || '';
        const planner = metadata.plan?.planner_source || '';
        const strategy = metadata.debug?.execution_strategy || '';
        const evidenceState = metadata.debug?.evidence_disposition?.state || '';
        metaDiv.innerHTML = [
            `<span>${escapeHTML(metadata.execution_mode.replaceAll('_', ' '))}</span>`,
            strategy ? `<span>${escapeHTML(strategy.replaceAll('_', ' '))}</span>` : '',
            evidenceState ? `<span>${escapeHTML(evidenceState.replaceAll('_', ' '))}</span>` : '',
            category ? `<span>${escapeHTML(category.replaceAll('_', ' '))}</span>` : '',
            planner ? `<span>${escapeHTML(planner.replaceAll('_', ' '))}</span>` : '',
        ].filter(Boolean).join('');
        contentDiv.prepend(metaDiv);
    }

    if (role === 'assistant' && metadata?.live_progress?.length) {
        const activity = document.createElement('details');
        activity.className = 'answer-activity';
        const summary = document.createElement('summary');
        summary.textContent = 'Recorded activity · ' + metadata.live_progress.length + (metadata.live_progress_truncated ? ' recent events (earlier events omitted)' : ' events');
        const log = document.createElement('ol');
        for (const event of metadata.live_progress) {
            log.appendChild(activityRow(event));
        }
        activity.append(summary, log);
        contentDiv.appendChild(activity);
    }

    if (sources?.length) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'sources-box';
        sourcesDiv.innerHTML = `<h4>Sources</h4><ul>${sources.map(source => {
            const label = source.source_file || source.evidence_path || source.source_path || source.source_id || 'source';
            const details = [source.chunk_type, source.paragraph, source.variable].filter(Boolean).join(' · ');
            return `<li>${escapeHTML(label)}${details ? `<br><small>${escapeHTML(details)}</small>` : ''}</li>`;
        }).join('')}</ul>`;
        contentDiv.appendChild(sourcesDiv);
    }

    if (role === 'assistant' && metadata?.debug) {
        const debugDiv = document.createElement('div');
        debugDiv.innerHTML = renderDebugDetails(metadata.debug, metadata.trace_id || '');
        if (debugDiv.firstElementChild) contentDiv.appendChild(debugDiv.firstElementChild);
    }

    msgDiv.append(avatar, contentDiv);
    chatHistory.appendChild(msgDiv);
    scrollChatToBottom();
}

function shuffleItems(items) {
    const shuffled = [...items];
    for (let index = shuffled.length - 1; index > 0; index -= 1) {
        const randomIndex = Math.floor(Math.random() * (index + 1));
        [shuffled[index], shuffled[randomIndex]] = [shuffled[randomIndex], shuffled[index]];
    }
    return shuffled;
}

function createWaitingExperience(requestId) {
    const element = document.createElement('div');
    element.className = 'message assistant loading waiting-experience';
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = 'AI';
    const content = document.createElement('div');
    content.className = 'content waiting-content';
    const header = document.createElement('div');
    header.className = 'waiting-header';
    const title = document.createElement('strong');
    title.textContent = 'Live RAG activity · detailed execution log';
    const elapsed = document.createElement('span');
    elapsed.className = 'waiting-elapsed';
    header.append(title, elapsed);
    const note = document.createElement('p');
    note.className = 'activity-note';
    note.textContent = 'Recorded actions, source lookups and validation—not private model reasoning. A running model call may have no new events.';
    const state = document.createElement('div');
    state.className = 'activity-state';
    state.setAttribute('role', 'status');
    state.textContent = 'Waiting for the server to receive this question…';
    const list = document.createElement('ol');
    list.className = 'activity-list';
    const stats = document.createElement('div');
    stats.className = 'activity-stats';
    stats.textContent = 'Observed: 0 files read · 0 tool calls · 0 model calls';
    content.append(header, note, state, stats, list);
    element.append(avatar, content);
    let destroyed = false, cursor = 0, timer = null, controller = null, inFlight = null;
    let failures = 0;
    const events = [];
    const files = new Set();
    let toolCalls = 0, modelCalls = 0, truncated = false;
    const started = Date.now();
    function show(event) {
        events.push(event);
        if (events.length > 200) { events.shift(); truncated = true; }
        const details = event.details || {};
        if (event.kind === 'file_read' && details.path) files.add(details.path);
        if (event.kind === 'tool_start') toolCalls++;
        if (event.kind === 'model_start') modelCalls++;
        stats.textContent = 'Observed: ' + files.size + ' files read · ' + toolCalls + ' tool calls · ' + modelCalls + ' model calls';
        const follow = !Number.isFinite(list.clientHeight) || list.scrollHeight - list.scrollTop - list.clientHeight < 60;
        list.appendChild(activityRow(event));
        while (list.childElementCount > 200) list.firstElementChild.remove();
        state.textContent = details.phase ? activityPhase(details.phase) + ' · ' + event.message
            : event.message + (details.path ? ' ' + details.path : '')
                + (details.source_file ? ' ' + details.source_file : '');
        if (follow) list.scrollTop = list.scrollHeight;
    }
    function refresh() {
        if (destroyed || inFlight) return inFlight;
        controller = new AbortController();
        const timeout = setTimeout(() => controller?.abort(), 5000);
        inFlight = (async () => {
            try {
                const data = await apiFetch('/api/chat/progress/' + encodeURIComponent(requestId) + '?after=' + cursor,
                    {signal: controller.signal});
                if (destroyed) return;
                failures = 0;
                if (data.truncated) {
                    truncated = true;
                    note.textContent = 'Earlier events were omitted; showing recent recorded activity, not private model reasoning. Indexed source views are distinguished from original-file reads.';
                }
                for (const event of data.events || []) {
                    if (event.sequence > cursor) { show(event); cursor = event.sequence; }
                }
            } catch (error) {
                if (!destroyed && ++failures >= 3) state.textContent = 'Activity feed temporarily unavailable. The answer request is still running.';
            } finally {
                clearTimeout(timeout);
                inFlight = null;
                if (!destroyed) timer = setTimeout(refresh, 1500);
            }
        })();
        return inFlight;
    }
    const clock = setInterval(() => { elapsed.textContent = Math.floor((Date.now() - started) / 1000) + 's'; }, 1000);
    refresh();
    return {
        element,
        snapshot() { return events.slice(); },
        truncated() { return truncated; },
        finish() { show({kind: 'client_result', message: 'Answer received.', elapsed_ms: Date.now() - started, details: {}}); },
        destroy() { destroyed = true; clearInterval(clock); clearTimeout(timer); controller?.abort(); }
    };
}

// Questions waiting to be asked, and the controller for the one in flight.
// Queued questions run strictly one at a time: each answer becomes context for
// the next, so overlapping them would resolve follow-ups against the wrong turn.
let pendingQuestions = [];
let runningRequest = null;

function splitQuestions(text) {
    return text
        .split(/\r\n|\n|\r/)
        .map(line => line.trim())
        .filter(Boolean);
}

function renderQueue() {
    queueList.innerHTML = '';
    if (!pendingQuestions.length) {
        queueBar.hidden = true;
        return;
    }
    queueBar.hidden = false;
    queueStatus.textContent = `${pendingQuestions.length} question${pendingQuestions.length === 1 ? '' : 's'} queued`;
    pendingQuestions.forEach(question => {
        const item = document.createElement('li');
        item.textContent = question;
        queueList.appendChild(item);
    });
}

function setRunning(isRunning) {
    sendButton.hidden = isRunning;
    stopButton.hidden = !isRunning;
    sendButton.disabled = isRunning;
}

async function askOne(text) {
    appendMessage('user', text);
    const requestId = newConversationId();
    const waiting = createWaitingExperience(requestId);
    chatHistory.appendChild(waiting.element);
    scrollChatToBottom();

    runningRequest = new AbortController();
    try {
        const data = await apiFetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text, request_id: requestId }),
            signal: runningRequest.signal,
        });
        observeConversationContext(data.session_context);
        if (data.execution_mode === 'cancelled') {
            appendMessage('assistant', 'Stopped. This answer was discarded and is not part of the chat memory.');
            return false;
        }
        waiting.finish?.();
        data.live_progress = waiting.snapshot?.() || [];
        data.live_progress_truncated = waiting.truncated?.() || false;
        appendMessage('assistant', data.answer, data.sources, data);
        return true;
    } catch (error) {
        if (error.name === 'AbortError') {
            appendMessage('assistant', 'Stopped. This answer was discarded and is not part of the chat memory.');
            return false;
        }
        appendMessage('assistant', `**Error:** ${error.message}`, [], {live_progress: waiting.snapshot?.() || []});
        return true;
    } finally {
        runningRequest = null;
        waiting.destroy();
        waiting.element.remove();
    }
}

async function drainQueue() {
    setRunning(true);
    try {
        while (pendingQuestions.length) {
            const next = pendingQuestions.shift();
            renderQueue();
            const keepGoing = await askOne(next);
            if (!keepGoing) {
                // A stop applies to the whole batch: the questions behind it were
                // queued expecting the earlier answers to be in context.
                const skipped = pendingQuestions.length;
                pendingQuestions = [];
                renderQueue();
                if (skipped) {
                    appendMessage('assistant', `${skipped} queued question${skipped === 1 ? '' : 's'} cancelled.`);
                }
                break;
            }
        }
    } finally {
        setRunning(false);
        renderQueue();
        chatInput.focus();
    }
}

async function enqueueQuestions(questions) {
    if (!questions.length) return;
    pendingQuestions = pendingQuestions.concat(questions);
    renderQueue();

    // Already draining means this batch was appended to the running queue and
    // will be picked up in turn, so a second loop must not start.
    if (runningRequest || sendButton.hidden) return;
    await drainQueue();
}

async function sendMessage() {
    const questions = splitQuestions(chatInput.value);
    if (!questions.length) return;
    chatInput.value = '';
    chatInput.style.height = 'auto';
    await enqueueQuestions(questions);
}

function updateBatchCount() {
    const count = splitQuestions(batchInput.value).length;
    batchCount.textContent = `${count} question${count === 1 ? '' : 's'}`;
    batchRunButton.disabled = count === 0;
}

document.getElementById('open-batch-btn').addEventListener('click', () => {
    updateBatchCount();
    batchDialog.showModal();
});
batchInput.addEventListener('input', updateBatchCount);
document.getElementById('batch-cancel-btn').addEventListener('click', () => batchDialog.close());
document.getElementById('batch-form').addEventListener('submit', event => {
    event.preventDefault();
    if (!batchDialog.open) return;
    const questions = splitQuestions(batchInput.value);
    if (!questions.length) return;
    batchInput.value = '';
    updateBatchCount();
    batchDialog.close();
    void enqueueQuestions(questions);
});

async function stopChat() {
    if (runningRequest) {
        runningRequest.abort();
    }
    try {
        await apiFetch('/api/chat/cancel', { method: 'POST' });
    } catch (error) {
        // The abort above already stopped the wait; a failed cancel only means
        // the discarded answer may still land in memory, which reset clears.
        console.warn('cancel request failed', error);
    }
}

async function resetChat() {
    pendingQuestions = [];
    renderQueue();
    if (runningRequest) {
        runningRequest.abort();
    }
    try {
        await apiFetch('/api/chat/reset', { method: 'POST' });
        chatSessionId = newConversationId();
        serverConversationContext = null;
        chatHistory.innerHTML = '';
        transcript = [];
        transcriptStartedAt = new Date().toISOString();
        appendMessage('assistant', 'Chat memory cleared. What should we inspect next?');
    } catch (error) {
        showModal('Chat Reset Failed', `<p>${escapeHTML(error.message)}</p>`);
    } finally {
        setRunning(false);
    }
}

chatInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = `${this.scrollHeight}px`;
});

document.addEventListener('DOMContentLoaded', () => {
    loadHealth();
    loadInbox();
});
