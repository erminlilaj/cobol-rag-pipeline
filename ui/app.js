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
const statusIndicator = document.getElementById('status-indicator');
const collectionLabel = document.getElementById('collection-label');
const treeFilter = document.getElementById('tree-filter');

let selectedInboxPaths = new Set();
let inboxTreeData = null;

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
            <h5>Evidence inspected (${evidence.length})</h5>
            ${evidence.length ? evidence.map(item => {
                const label = item.source_file || item.source_id || `Evidence ${item.rank || ''}`;
                const descriptors = [item.chunk_type, item.program, item.entity_key].filter(Boolean).join(' · ');
                const score = item.score === null || item.score === undefined ? '' : ` · score ${Number(item.score).toFixed(4)}`;
                return `<details class="debug-nested">
                    <summary>${escapeHTML(label)}${score}</summary>
                    ${descriptors ? `<p class="debug-muted">${escapeHTML(descriptors)}</p>` : ''}
                    <pre>${escapeHTML(item.excerpt || 'No text excerpt available.')}</pre>
                </details>`;
            }).join('') : '<p class="debug-muted">No evidence record was attached to this route.</p>'}
        </section>`;

    const planBlock = `
        <section class="debug-section">
            <h5>Query plan</h5>
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
    const response = await fetch(url, options);
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

function appendMessage(role, content, sources = [], metadata = null) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = role === 'user' ? 'U' : 'AI';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'content';
    contentDiv.innerHTML = renderMarkdownLite(content);

    if (role === 'assistant' && metadata?.execution_mode) {
        const metaDiv = document.createElement('div');
        metaDiv.className = 'answer-meta';
        const category = metadata.plan?.category || metadata.route || '';
        const planner = metadata.plan?.planner_source || '';
        metaDiv.innerHTML = [
            `<span>${escapeHTML(metadata.execution_mode.replaceAll('_', ' '))}</span>`,
            category ? `<span>${escapeHTML(category.replaceAll('_', ' '))}</span>` : '',
            planner ? `<span>${escapeHTML(planner.replaceAll('_', ' '))}</span>` : '',
        ].filter(Boolean).join('');
        contentDiv.prepend(metaDiv);
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

function createWaitingExperience() {
    const element = document.createElement('div');
    element.className = 'message assistant loading waiting-experience';

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = 'AI';

    const content = document.createElement('div');
    content.className = 'content waiting-content';

    const header = document.createElement('div');
    header.className = 'waiting-header';
    const headerCopy = document.createElement('div');
    const title = document.createElement('strong');
    title.textContent = 'Working on your answer';
    const subtitle = document.createElement('span');
    subtitle.textContent = 'Play a quick game while evidence is being prepared.';
    headerCopy.append(title, subtitle);
    const elapsed = document.createElement('span');
    elapsed.className = 'waiting-elapsed';
    elapsed.textContent = '0s';
    elapsed.setAttribute('aria-label', 'Elapsed time');
    header.append(headerCopy, elapsed);

    const tabs = document.createElement('div');
    tabs.className = 'waiting-game-tabs';
    tabs.setAttribute('role', 'tablist');
    tabs.setAttribute('aria-label', 'Waiting games');

    const panel = document.createElement('div');
    panel.className = 'waiting-game-panel';
    panel.setAttribute('aria-live', 'polite');

    content.append(header, tabs, panel);
    element.append(avatar, content);

    const gameState = {
        activeGame: 'bug',
        bugScore: 0,
        bugTarget: Math.floor(Math.random() * 12),
        memoryDeck: shuffleItems(['MOVE', 'MOVE', 'PERFORM', 'PERFORM', 'LINK', 'LINK', 'XCTL', 'XCTL']),
        memoryRevealed: new Set(),
        memoryMatched: new Set(),
        memoryLocked: false,
        memoryMoves: 0,
        destroyed: false,
        timeouts: new Set(),
    };

    function schedule(callback, delay) {
        const timeout = setTimeout(() => {
            gameState.timeouts.delete(timeout);
            if (!gameState.destroyed) callback();
        }, delay);
        gameState.timeouts.add(timeout);
    }

    function renderTabs() {
        tabs.innerHTML = '';
        [
            ['bug', 'Bug Hunt'],
            ['memory', 'COBOL Match'],
        ].forEach(([game, label]) => {
            const button = document.createElement('button');
            const selected = gameState.activeGame === game;
            button.type = 'button';
            button.className = `waiting-game-tab${selected ? ' active' : ''}`;
            button.textContent = label;
            button.setAttribute('role', 'tab');
            button.setAttribute('aria-selected', String(selected));
            button.addEventListener('click', () => {
                gameState.activeGame = game;
                renderTabs();
                renderGame();
            });
            tabs.appendChild(button);
        });
    }

    function renderBugHunt() {
        panel.innerHTML = '';
        const status = document.createElement('div');
        status.className = 'waiting-game-status';
        status.innerHTML = `<span>Catch the moving bug.</span><strong>Score: ${gameState.bugScore}</strong>`;

        const grid = document.createElement('div');
        grid.className = 'bug-grid';
        grid.setAttribute('aria-label', 'Bug hunt grid');
        for (let index = 0; index < 12; index += 1) {
            const cell = document.createElement('button');
            const isTarget = index === gameState.bugTarget;
            cell.type = 'button';
            cell.className = `bug-cell${isTarget ? ' target' : ''}`;
            cell.textContent = isTarget ? '◆' : '·';
            cell.setAttribute('aria-label', isTarget ? 'Catch the bug' : 'Empty cell');
            if (isTarget) {
                cell.addEventListener('click', () => {
                    gameState.bugScore += 1;
                    moveBug();
                });
            }
            grid.appendChild(cell);
        }
        panel.append(status, grid);
    }

    function moveBug() {
        let nextTarget = Math.floor(Math.random() * 12);
        if (nextTarget === gameState.bugTarget) {
            nextTarget = (nextTarget + 1) % 12;
        }
        gameState.bugTarget = nextTarget;
        if (gameState.activeGame === 'bug') renderBugHunt();
    }

    function renderMemory() {
        panel.innerHTML = '';
        const status = document.createElement('div');
        status.className = 'waiting-game-status';
        const pairs = gameState.memoryMatched.size / 2;
        const message = pairs === 4 ? 'All pairs matched!' : 'Match the COBOL keywords.';
        status.innerHTML = `<span>${message}</span><strong>${pairs}/4 pairs · ${gameState.memoryMoves} moves</strong>`;

        const grid = document.createElement('div');
        grid.className = 'memory-grid';
        grid.setAttribute('aria-label', 'COBOL keyword memory cards');
        gameState.memoryDeck.forEach((keyword, index) => {
            const card = document.createElement('button');
            const visible = gameState.memoryRevealed.has(index) || gameState.memoryMatched.has(index);
            const matched = gameState.memoryMatched.has(index);
            card.type = 'button';
            card.className = `memory-card${visible ? ' revealed' : ''}${matched ? ' matched' : ''}`;
            card.textContent = visible ? keyword : '?';
            card.disabled = matched || gameState.memoryLocked;
            card.setAttribute('aria-label', visible ? keyword : 'Hidden keyword card');
            card.addEventListener('click', () => revealMemoryCard(index));
            grid.appendChild(card);
        });
        panel.append(status, grid);
    }

    function revealMemoryCard(index) {
        if (
            gameState.memoryLocked ||
            gameState.memoryMatched.has(index) ||
            gameState.memoryRevealed.has(index)
        ) return;

        gameState.memoryRevealed.add(index);
        const revealed = Array.from(gameState.memoryRevealed);
        if (revealed.length === 2) {
            gameState.memoryMoves += 1;
            const [first, second] = revealed;
            if (gameState.memoryDeck[first] === gameState.memoryDeck[second]) {
                gameState.memoryMatched.add(first);
                gameState.memoryMatched.add(second);
                gameState.memoryRevealed.clear();
            } else {
                gameState.memoryLocked = true;
                schedule(() => {
                    gameState.memoryRevealed.clear();
                    gameState.memoryLocked = false;
                    if (gameState.activeGame === 'memory') renderMemory();
                }, 700);
            }
        }
        renderMemory();
    }

    function renderGame() {
        if (gameState.activeGame === 'memory') {
            renderMemory();
        } else {
            renderBugHunt();
        }
    }

    renderTabs();
    renderGame();

    let seconds = 0;
    const elapsedInterval = setInterval(() => {
        seconds += 1;
        elapsed.textContent = `${seconds}s`;
    }, 1000);
    const bugInterval = setInterval(() => {
        if (!element.isConnected && seconds > 0) {
            destroy();
            return;
        }
        moveBug();
    }, 1350);

    function destroy() {
        if (gameState.destroyed) return;
        gameState.destroyed = true;
        clearInterval(elapsedInterval);
        clearInterval(bugInterval);
        gameState.timeouts.forEach(timeout => clearTimeout(timeout));
        gameState.timeouts.clear();
    }

    return { element, destroy };
}

// Questions waiting to be asked, and the controller for the one in flight.
// Queued questions run strictly one at a time: each answer becomes context for
// the next, so overlapping them would resolve follow-ups against the wrong turn.
let pendingQuestions = [];
let runningRequest = null;

function splitQuestions(text) {
    return text
        .split('\n')
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
    const waiting = createWaitingExperience();
    chatHistory.appendChild(waiting.element);
    scrollChatToBottom();

    runningRequest = new AbortController();
    try {
        const data = await apiFetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text }),
            signal: runningRequest.signal,
        });
        appendMessage('assistant', data.answer, data.sources, data);
        return true;
    } catch (error) {
        if (error.name === 'AbortError') {
            appendMessage('assistant', 'Stopped. This answer was discarded and is not part of the chat memory.');
            return false;
        }
        appendMessage('assistant', `**Error:** ${error.message}`);
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

async function sendMessage() {
    const questions = splitQuestions(chatInput.value);
    if (!questions.length) return;

    chatInput.value = '';
    chatInput.style.height = 'auto';
    pendingQuestions.push(...questions);
    renderQueue();

    // Already draining means this batch was appended to the running queue and
    // will be picked up in turn, so a second loop must not start.
    if (runningRequest || sendButton.hidden) return;
    await drainQueue();
}

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
        chatHistory.innerHTML = '';
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
