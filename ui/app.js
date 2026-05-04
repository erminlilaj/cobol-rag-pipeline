// UI Interactions
function toggleAccordion(element) {
    const parent = element.parentElement;
    parent.classList.toggle('active');
}

function handleEnter(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

// Modal handling
const modal = document.getElementById('modal');
const modalTitle = document.getElementById('modal-title');
const modalBody = document.getElementById('modal-body');

function showModal(title, contentHTML) {
    modalTitle.textContent = title;
    modalBody.innerHTML = contentHTML;
    modal.style.display = 'flex';
}

function closeModal() {
    modal.style.display = 'none';
}

window.onclick = function(event) {
    if (event.target == modal) {
        closeModal();
    }
}

// Global state for inbox selection
let selectedInboxPaths = new Set();

// Render Inbox Tree
function renderTree(node) {
    const div = document.createElement('div');
    div.className = 'tree-node';
    
    const content = document.createElement('div');
    content.className = 'tree-node-content';
    
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.value = node.path;
    // Add logic to check/uncheck children if it's a directory
    checkbox.onchange = (e) => {
        handleCheckboxChange(e.target.checked, node, div);
    };
    
    const icon = document.createElement('span');
    icon.innerHTML = node.is_dir ? '&#128193;' : '&#128196;';
    
    const label = document.createElement('span');
    label.textContent = node.name;
    
    content.appendChild(checkbox);
    content.appendChild(icon);
    content.appendChild(label);
    div.appendChild(content);
    
    if (node.is_dir && node.children && node.children.length > 0) {
        const childrenContainer = document.createElement('div');
        childrenContainer.className = 'tree-children';
        node.children.forEach(child => {
            childrenContainer.appendChild(renderTree(child));
        });
        div.appendChild(childrenContainer);
    }
    
    return div;
}

function handleCheckboxChange(checked, node, element) {
    // Update Set
    if (checked) {
        selectedInboxPaths.add(node.path);
    } else {
        selectedInboxPaths.delete(node.path);
    }
    
    // If it's a directory, recursively check/uncheck all children visually and in state
    if (node.is_dir) {
        const checkboxes = element.querySelectorAll('.tree-children input[type="checkbox"]');
        checkboxes.forEach(cb => {
            cb.checked = checked;
            if (checked) {
                selectedInboxPaths.add(cb.value);
            } else {
                selectedInboxPaths.delete(cb.value);
            }
        });
    }
}

// API Calls
async function loadInbox() {
    try {
        const res = await fetch('/api/inbox');
        if (res.ok) {
            const data = await res.json();
            const container = document.getElementById('inbox-tree');
            container.innerHTML = '';
            selectedInboxPaths.clear();
            container.appendChild(renderTree(data));
        }
    } catch (e) {
        console.error("Failed to load inbox", e);
    }
}

async function syncSelected() {
    const btn = document.getElementById('btn-sync');
    btn.textContent = 'Syncing...';
    btn.disabled = true;
    
    try {
        const payload = { paths: selectedInboxPaths.size > 0 ? Array.from(selectedInboxPaths) : null };
        const res = await fetch('/api/sync', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        let html = `<p><strong>Collection:</strong> ${data.collection}</p>
                    <p><strong>Processed:</strong> ${data.documents_processed}</p>
                    <p><strong>Added:</strong> ${data.added}</p>
                    <p><strong>Updated:</strong> ${data.updated}</p>
                    <p><strong>Skipped:</strong> ${data.skipped}</p>`;
        showModal('Sync Complete', html);
    } catch (e) {
        showModal('Error', `<p>Failed to sync: ${e.message}</p>`);
    } finally {
        btn.textContent = 'Sync Selected';
        btn.disabled = false;
    }
}

async function fetchConfig() {
    try {
        const res = await fetch('/api/config');
        const data = await res.json();
        showModal('Configuration', `<pre>${JSON.stringify(data, null, 2)}</pre>`);
    } catch (e) {
        showModal('Error', `<p>Failed to fetch config.</p>`);
    }
}

async function fetchIndexInfo() {
    try {
        const res = await fetch('/api/index-info');
        const data = await res.json();
        showModal('Index Info', `<pre>${JSON.stringify(data, null, 2)}</pre>`);
    } catch (e) {
        showModal('Error', `<p>Failed to fetch index info.</p>`);
    }
}

async function resetCollection() {
    if (!confirm('Are you sure you want to reset the Chroma DB?')) return;
    try {
        const res = await fetch('/api/reset', { method: 'POST' });
        const data = await res.json();
        showModal('Reset Collection', `<p>${data.message}</p>`);
    } catch (e) {
        showModal('Error', `<p>Failed to reset collection.</p>`);
    }
}

// Chat functionality
const chatHistory = document.getElementById('chat-history');
const chatInput = document.getElementById('chat-input');

function appendMessage(role, content, sources = []) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = role === 'user' ? 'U' : 'AI';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'content';
    // Use marked to parse markdown
    contentDiv.innerHTML = marked.parse(content);
    
    if (sources && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'sources-box';
        sourcesDiv.innerHTML = `<h4>Sources</h4><ul>` + 
            sources.map(s => `<li>${s.source_path} (Score: ${s.score ? s.score.toFixed(3) : 'N/A'})</li>`).join('') +
            `</ul>`;
        contentDiv.appendChild(sourcesDiv);
    }
    
    msgDiv.appendChild(avatar);
    msgDiv.appendChild(contentDiv);
    
    chatHistory.appendChild(msgDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;
    
    appendMessage('user', text);
    chatInput.value = '';
    
    // Add loading indicator
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message assistant loading';
    loadingDiv.innerHTML = `<div class="avatar">AI</div><div class="content">Thinking...</div>`;
    chatHistory.appendChild(loadingDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
    
    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text })
        });
        
        loadingDiv.remove();
        
        if (res.ok) {
            const data = await res.json();
            appendMessage('assistant', data.answer, data.sources);
        } else {
            const err = await res.json();
            appendMessage('assistant', `**Error:** ${err.detail || 'Failed to get answer'}`);
        }
    } catch (e) {
        loadingDiv.remove();
        appendMessage('assistant', `**Error:** Network request failed.`);
    }
}

async function resetChat() {
    try {
        await fetch('/api/chat/reset', { method: 'POST' });
        chatHistory.innerHTML = `<div class="message assistant"><div class="avatar">AI</div><div class="content">Chat memory cleared. How can I help?</div></div>`;
    } catch (e) {
        alert("Failed to reset chat");
    }
}

// Auto-resize textarea
chatInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadInbox();
});
