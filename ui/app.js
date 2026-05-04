const modal = document.getElementById('modal');
const modalTitle = document.getElementById('modal-title');
const modalBody = document.getElementById('modal-body');
const chatHistory = document.getElementById('chat-history');
const chatInput = document.getElementById('chat-input');
const sendButton = document.getElementById('send-btn');
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

function appendMessage(role, content, sources = []) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = role === 'user' ? 'U' : 'AI';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'content';
    contentDiv.innerHTML = renderMarkdownLite(content);

    if (sources?.length) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'sources-box';
        sourcesDiv.innerHTML = `<h4>Sources</h4><ul>${sources.map(source => {
            const label = source.source_path || source.source_id || 'source';
            const score = source.score == null ? 'N/A' : Number(source.score).toFixed(3);
            return `<li>${escapeHTML(label)} · score ${escapeHTML(score)}</li>`;
        }).join('')}</ul>`;
        contentDiv.appendChild(sourcesDiv);
    }

    msgDiv.append(avatar, contentDiv);
    chatHistory.appendChild(msgDiv);
    scrollChatToBottom();
}

async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text || sendButton.disabled) return;

    appendMessage('user', text);
    chatInput.value = '';
    chatInput.style.height = 'auto';
    sendButton.disabled = true;

    const loading = document.createElement('div');
    loading.className = 'message assistant loading';
    loading.innerHTML = '<div class="avatar">AI</div><div class="content">Thinking...</div>';
    chatHistory.appendChild(loading);
    scrollChatToBottom();

    try {
        const data = await apiFetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text }),
        });
        loading.remove();
        appendMessage('assistant', data.answer, data.sources);
    } catch (error) {
        loading.remove();
        appendMessage('assistant', `**Error:** ${error.message}`);
    } finally {
        sendButton.disabled = false;
        chatInput.focus();
    }
}

async function resetChat() {
    try {
        await apiFetch('/api/chat/reset', { method: 'POST' });
        chatHistory.innerHTML = '';
        appendMessage('assistant', 'Chat memory cleared. What should we inspect next?');
    } catch (error) {
        showModal('Chat Reset Failed', `<p>${escapeHTML(error.message)}</p>`);
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
