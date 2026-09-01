// Global App State
const state = {
  activeTab: 'chat',
  documents: [],
  chunks: [],
  stats: {},
  apiKey: localStorage.getItem('nexus_gemini_key') || '',
  settings: {
    topK: 4,
    chunkSize: 500,
    chunkOverlap: 100,
  },
  metrics: {
    retrievalTimes: [],
    genTimes: [],
  }
};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
  initApp();
});

async function initApp() {
  loadStoredSettings();
  setupEventListeners();
  await refreshAll();
  updateApiKeyStatusUI();
}

function loadStoredSettings() {
  if (state.apiKey) {
    document.getElementById('setting-api-key').value = state.apiKey;
  }
}

function updateApiKeyStatusUI() {
  const btnLabel = document.getElementById('api-key-status-label');
  const engineLabel = document.getElementById('current-engine-label');
  
  if (state.apiKey) {
    btnLabel.innerText = 'Gemini API Connected';
    engineLabel.innerText = 'Google Gemini 3.6 Flash';
  } else {
    btnLabel.innerText = 'Configure Gemini API';
    engineLabel.innerText = 'Local Extractive (Offline Ready)';
  }
}

function setupEventListeners() {
  // Drag and Drop
  const dropzone = document.getElementById('dropzone');
  if (dropzone) {
    ['dragenter', 'dragover'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropzone.classList.add('drag-over');
      }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropzone.classList.remove('drag-over');
      }, false);
    });

    dropzone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      const files = dt.files;
      if (files && files.length > 0) {
        uploadFiles(files);
      }
    });
  }
}

// Tab Switching
function switchTab(tabId) {
  state.activeTab = tabId;
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === tabId);
  });
  document.querySelectorAll('.tab-view').forEach(view => {
    view.classList.toggle('active', view.id === `tab-${tabId}`);
  });

  if (tabId === 'explorer') {
    renderChunksExplorer();
  } else if (tabId === 'analytics') {
    renderAnalytics();
  }
}

// Data Fetching & Sync
async function refreshAll() {
  await Promise.all([fetchStats(), fetchDocuments(), fetchChunks()]);
}

async function fetchStats() {
  try {
    const res = await fetch('/api/stats');
    if (res.ok) {
      state.stats = await res.json();
      updateStatsUI();
    }
  } catch (err) {
    console.error('Error fetching stats:', err);
  }
}

async function fetchDocuments() {
  try {
    const res = await fetch('/api/documents');
    if (res.ok) {
      state.documents = await res.json();
      renderDocumentList();
    }
  } catch (err) {
    console.error('Error fetching documents:', err);
  }
}

async function fetchChunks() {
  try {
    const res = await fetch('/api/chunks');
    if (res.ok) {
      const data = await res.json();
      state.chunks = data.chunks || [];
    }
  } catch (err) {
    console.error('Error fetching chunks:', err);
  }
}

function updateStatsUI() {
  const docChunkCount = document.getElementById('doc-chunk-count');
  if (docChunkCount) {
    docChunkCount.innerText = `${state.stats.total_chunks || 0} Chunks`;
  }
  const docCountBadge = document.getElementById('doc-count-badge');
  if (docCountBadge) {
    docCountBadge.innerText = `${state.stats.total_documents || 0} Files`;
  }
}

function renderDocumentList() {
  const listEl = document.getElementById('document-list');
  if (!listEl) return;

  if (!state.documents || state.documents.length === 0) {
    listEl.innerHTML = `
      <div class="empty-docs-placeholder" id="empty-docs-msg">
        <i class="fa-regular fa-folder-open"></i>
        <p>No documents uploaded yet.</p>
        <span>Upload PDF/TXT documents or load sample documents below to begin.</span>
      </div>
    `;
    return;
  }

  listEl.innerHTML = state.documents.map(doc => `
    <div class="doc-card">
      <div class="doc-info">
        <i class="fa-solid ${getFileIcon(doc.file_type)} doc-icon"></i>
        <div class="doc-details">
          <h4 title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</h4>
          <p>${formatBytes(doc.file_size)} • ${doc.chunk_count} chunks</p>
        </div>
      </div>
      <div class="doc-actions">
        <button class="del-doc-btn" onclick="deleteDocument('${doc.id}')" title="Delete document">
          <i class="fa-solid fa-trash"></i>
        </button>
      </div>
    </div>
  `).join('');
}

function getFileIcon(type) {
  switch (type.toUpperCase()) {
    case 'PDF': return 'fa-file-pdf';
    case 'DOCX':
    case 'DOC': return 'fa-file-word';
    case 'TXT': return 'fa-file-lines';
    case 'MD': return 'fa-file-code';
    default: return 'fa-file';
  }
}

function formatBytes(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

// File Upload Handlers
function triggerFileInput() {
  document.getElementById('file-input').click();
}

function handleFileSelect(event) {
  const files = event.target.files;
  if (files && files.length > 0) {
    uploadFiles(files);
  }
  event.target.value = '';
}

async function uploadFiles(files) {
  for (let file of files) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('chunk_size', state.settings.chunkSize);
    formData.append('chunk_overlap', state.settings.chunkOverlap);

    showToast(`Uploading and chunking ${file.name}...`, 'info');

    try {
      const res = await fetch('/api/documents/upload', {
        method: 'POST',
        body: formData,
      });

      if (res.ok) {
        const data = await res.json();
        showToast(`Indexed ${file.name} into ${data.chunks_created} chunks!`, 'success');
        await refreshAll();
      } else {
        const err = await res.json();
        showToast(`Failed to upload ${file.name}: ${err.detail}`, 'error');
      }
    } catch (err) {
      showToast(`Error uploading ${file.name}`, 'error');
    }
  }
}

async function deleteDocument(docId) {
  try {
    const res = await fetch(`/api/documents/${docId}`, { method: 'DELETE' });
    if (res.ok) {
      showToast('Document removed', 'info');
      await refreshAll();
    }
  } catch (err) {
    showToast('Failed to delete document', 'error');
  }
}

async function clearKnowledgeBase() {
  if (!confirm('Are you sure you want to clear all uploaded documents and vectors?')) return;
  try {
    const res = await fetch('/api/documents/clear', { method: 'POST' });
    if (res.ok) {
      showToast('Knowledge Base reset', 'info');
      await refreshAll();
      renderChunksExplorer();
    }
  } catch (err) {
    showToast('Failed to clear knowledge base', 'error');
  }
}

// Load Demo Sample Documents
async function loadSampleDocs() {
  showToast('Loading demo sample documents...', 'info');
  const sample1Content = `Generative AI & RAG Architecture Overview:
Retrieval-Augmented Generation (RAG) combines dense vector retrieval with Large Language Models (LLMs).
Key Workflow:
1. Document Ingestion: Documents (PDFs, text) are parsed and split into overlapping chunks (e.g. 500 characters with 100 char overlap).
2. Embedding Generation: Each chunk is mapped to a vector embedding space using TF-IDF or transformer models.
3. Vector Indexing: Embeddings are stored in a vector database for high-speed cosine similarity retrieval.
4. Prompt Augmentation: User queries retrieve Top-K matching chunks which are injected into the LLM system prompt as ground-truth context.
5. Citation & Verification: Responses cite exact document titles and page numbers to prevent hallucination.`;

  const sample2Content = `Nexus Corp Employee Operations & Policy Guide:
Working Hours: Standard working hours are 9:00 AM to 5:00 PM EST, Monday through Friday.
Remote Work Policy: Employees are eligible for hybrid work up to 3 days per week with manager approval.
Security & Data Privacy: All proprietary software code, API credentials, and internal customer records must be stored on encrypted company devices.
Expense Reimbursement: Travel expenses and client meals can be claimed up to $150 daily via the internal portal within 14 days.`;

  const file1 = new File([sample1Content], "Generative_AI_Overview.txt", { type: "text/plain" });
  const file2 = new File([sample2Content], "Company_Policies_Handbook.txt", { type: "text/plain" });

  await uploadFiles([file1, file2]);
}

// Chat UI Handlers
function handleKeyDown(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    document.getElementById('chat-form').dispatchEvent(new Event('submit'));
  }
}

function useSuggestedQuery(btn) {
  const query = btn.innerText;
  document.getElementById('query-input').value = query;
  document.getElementById('chat-form').dispatchEvent(new Event('submit'));
}

async function submitQuery(event) {
  event.preventDefault();
  const inputEl = document.getElementById('query-input');
  const query = inputEl.value.trim();
  if (!query) return;

  inputEl.value = '';
  const chatFeed = document.getElementById('chat-feed');

  // Hide welcome card if present
  const welcomeCard = document.querySelector('.chat-welcome-card');
  if (welcomeCard) welcomeCard.style.display = 'none';

  // Render User Message
  renderUserMessage(query);

  // Render AI Loading Skeleton
  const loadingBubble = renderAILoadingMessage();
  chatFeed.scrollTop = chatFeed.scrollHeight;

  try {
    const res = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query,
        top_k: parseInt(state.settings.topK),
        api_key: state.apiKey,
      }),
    });

    if (res.ok) {
      const data = await res.json();
      
      // Update latency stats
      state.metrics.retrievalTimes.push(data.retrieval_time_ms);
      state.metrics.genTimes.push(data.generation_time_ms);

      // Replace loading bubble with full response
      loadingBubble.remove();
      renderAIMessage(data);
    } else {
      const err = await res.json();
      loadingBubble.remove();
      renderAIErrorMessage(err.detail || 'Error processing request');
    }
  } catch (err) {
    loadingBubble.remove();
    renderAIErrorMessage('Failed to connect to backend server');
  }

  chatFeed.scrollTop = chatFeed.scrollHeight;
}

function renderUserMessage(query) {
  const chatFeed = document.getElementById('chat-feed');
  const msgRow = document.createElement('div');
  msgRow.className = 'message-row user-row';
  msgRow.innerHTML = `
    <div class="msg-bubble">
      ${escapeHtml(query)}
    </div>
    <div class="avatar"><i class="fa-solid fa-user"></i></div>
  `;
  chatFeed.appendChild(msgRow);
}

function renderAILoadingMessage() {
  const chatFeed = document.getElementById('chat-feed');
  const msgRow = document.createElement('div');
  msgRow.className = 'message-row ai-row';
  msgRow.innerHTML = `
    <div class="avatar"><i class="fa-solid fa-robot"></i></div>
    <div class="msg-bubble">
      <div class="meta-bar">
        <span><i class="fa-solid fa-spinner fa-spin"></i> Searching vector space & synthesizing context...</span>
      </div>
    </div>
  `;
  chatFeed.appendChild(msgRow);
  return msgRow;
}

function renderAIMessage(data) {
  const chatFeed = document.getElementById('chat-feed');
  const msgRow = document.createElement('div');
  msgRow.className = 'message-row ai-row';

  const sourcesHtml = data.sources && data.sources.length > 0 ? `
    <div class="sources-accordion">
      <div class="sources-header" onclick="toggleSources(this)">
        <span><i class="fa-solid fa-quote-left"></i> Verified Sources (${data.sources.length} cited passages)</span>
        <i class="fa-solid fa-chevron-down"></i>
      </div>
      <div class="sources-content">
        ${data.sources.map((s, idx) => `
          <div class="source-item">
            <div class="source-title">
              <span>[${idx + 1}] ${escapeHtml(s.doc_name)} (Page ${s.page_number})</span>
              <span class="score-tag ${s.similarity_score > 0.4 ? 'score-high' : 'score-med'}">
                ${Math.round(s.similarity_score * 100)}% match
              </span>
            </div>
            <div class="source-text">${escapeHtml(s.content)}</div>
          </div>
        `).join('')}
      </div>
    </div>
  ` : '';

  msgRow.innerHTML = `
    <div class="avatar"><i class="fa-solid fa-robot"></i></div>
    <div class="msg-bubble">
      <div class="meta-bar">
        <span><i class="fa-solid fa-microchip"></i> ${data.model_used}</span>
        <span>•</span>
        <span><i class="fa-solid fa-clock"></i> Total: ${data.total_time_ms}ms (Retrieval: ${data.retrieval_time_ms}ms)</span>
      </div>
      <div class="answer-content">${formatMarkdownText(data.answer)}</div>
      ${sourcesHtml}
    </div>
  `;
  chatFeed.appendChild(msgRow);
}

function renderAIErrorMessage(msg) {
  const chatFeed = document.getElementById('chat-feed');
  const msgRow = document.createElement('div');
  msgRow.className = 'message-row ai-row';
  msgRow.innerHTML = `
    <div class="avatar"><i class="fa-solid fa-triangle-exclamation"></i></div>
    <div class="msg-bubble" style="border-color: var(--danger);">
      <div class="answer-content" style="color: #fca5a5;">⚠️ ${escapeHtml(msg)}</div>
    </div>
  `;
  chatFeed.appendChild(msgRow);
}

function toggleSources(headerEl) {
  const content = headerEl.nextElementSibling;
  const icon = headerEl.querySelector('.fa-chevron-down, .fa-chevron-up');
  if (content.style.display === 'none') {
    content.style.display = 'flex';
    if (icon) icon.className = 'fa-solid fa-chevron-up';
  } else {
    content.style.display = 'none';
    if (icon) icon.className = 'fa-solid fa-chevron-down';
  }
}

// Vector Explorer Handlers
function renderChunksExplorer() {
  const grid = document.getElementById('chunks-grid');
  const countLabel = document.getElementById('showing-chunks-count');
  
  if (!state.chunks || state.chunks.length === 0) {
    grid.innerHTML = '<p class="text-dim" style="grid-column: 1/-1; text-align: center; padding: 40px;">No vector chunks currently indexed in storage.</p>';
    countLabel.innerText = 'Showing 0 Chunks';
    return;
  }

  countLabel.innerText = `Showing ${state.chunks.length} Chunks`;
  grid.innerHTML = state.chunks.map((c, idx) => `
    <div class="chunk-card">
      <div class="chunk-header">
        <span class="chunk-tag">Chunk #${c.chunk_index + 1}</span>
        <span class="text-dim">${escapeHtml(c.doc_name)} (Page ${c.page_number})</span>
      </div>
      <div class="chunk-body">${escapeHtml(c.content)}</div>
      <div class="text-dim" style="font-size: 10px;">ID: ${c.id}</div>
    </div>
  `).join('');
}

async function testVectorSearch() {
  const queryInput = document.getElementById('explorer-search');
  const query = queryInput.value.trim();
  if (!query) return;

  try {
    const res = await fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query, top_k: 6 }),
    });

    if (res.ok) {
      const data = await res.json();
      const grid = document.getElementById('chunks-grid');
      const countLabel = document.getElementById('showing-chunks-count');
      countLabel.innerText = `Top ${data.results.length} Similarity Matches for "${query}"`;

      grid.innerHTML = data.results.map((c) => `
        <div class="chunk-card" style="border-color: rgba(99, 102, 241, 0.4);">
          <div class="chunk-header">
            <span class="chunk-tag" style="background: rgba(16, 185, 129, 0.2); color: #34d399;">
              Match Score: ${Math.round(c.similarity_score * 100)}%
            </span>
            <span class="text-dim">${escapeHtml(c.doc_name)}</span>
          </div>
          <div class="chunk-body">${escapeHtml(c.content)}</div>
        </div>
      `).join('');
    }
  } catch (err) {
    showToast('Vector search failed', 'error');
  }
}

function filterChunks() {
  const query = document.getElementById('explorer-search').value.toLowerCase();
  if (!query) {
    renderChunksExplorer();
    return;
  }
  const filtered = state.chunks.filter(c => 
    c.content.toLowerCase().includes(query) || c.doc_name.toLowerCase().includes(query)
  );
  const grid = document.getElementById('chunks-grid');
  document.getElementById('showing-chunks-count').innerText = `Filtered ${filtered.length} Chunks`;

  grid.innerHTML = filtered.map(c => `
    <div class="chunk-card">
      <div class="chunk-header">
        <span class="chunk-tag">Chunk #${c.chunk_index + 1}</span>
        <span class="text-dim">${escapeHtml(c.doc_name)}</span>
      </div>
      <div class="chunk-body">${escapeHtml(c.content)}</div>
    </div>
  `).join('');
}

// Analytics Handlers
function renderAnalytics() {
  document.getElementById('metric-docs').innerText = state.documents.length;
  document.getElementById('metric-chunks').innerText = state.chunks.length;

  const rTimes = state.metrics.retrievalTimes;
  const avgRet = rTimes.length ? Math.round(rTimes.reduce((a, b) => a + b, 0) / rTimes.length) : 0;
  document.getElementById('metric-retrieval-time').innerText = `${avgRet} ms`;

  const gTimes = state.metrics.genTimes;
  const avgGen = gTimes.length ? Math.round(gTimes.reduce((a, b) => a + b, 0) / gTimes.length) : 0;
  document.getElementById('metric-gen-time').innerText = `${avgGen} ms`;
}

// Settings Handlers
function saveApiKey() {
  const keyInput = document.getElementById('setting-api-key').value.trim();
  state.apiKey = keyInput;
  if (keyInput) {
    localStorage.setItem('nexus_gemini_key', keyInput);
    showToast('Gemini API key saved successfully!', 'success');
  } else {
    localStorage.removeItem('nexus_gemini_key');
    showToast('Gemini API key cleared', 'info');
  }
  updateApiKeyStatusUI();
}

function updateRangeLabel(elementId, value) {
  document.getElementById(elementId).innerText = value;
}

function saveSettings() {
  state.settings.topK = document.getElementById('setting-top-k').value;
  state.settings.chunkSize = document.getElementById('setting-chunk-size').value;
  state.settings.chunkOverlap = document.getElementById('setting-chunk-overlap').value;

  fetch('/api/settings', {
    method: 'POST',
    body: new URLSearchParams({
      chunk_size: state.settings.chunkSize,
      chunk_overlap: state.settings.chunkOverlap,
    })
  });

  showToast('Retrieval hyperparameters updated', 'success');
}

function togglePasswordVisibility(inputId) {
  const input = document.getElementById(inputId);
  const eye = document.getElementById(`${inputId}-eye`);
  if (input.type === 'password') {
    input.type = 'text';
    eye.className = 'fa-solid fa-eye-slash';
  } else {
    input.type = 'password';
    eye.className = 'fa-solid fa-eye';
  }
}

// Helper Utilities
function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <i class="fa-solid ${type === 'success' ? 'fa-circle-check' : type === 'error' ? 'fa-circle-exclamation' : 'fa-circle-info'}"></i>
    <span>${escapeHtml(message)}</span>
  `;
  container.appendChild(toast);

  setTimeout(() => {
    toast.remove();
  }, 4000);
}

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

function formatMarkdownText(text) {
  // Simple regex for bold, inline code, paragraphs
  let formatted = escapeHtml(text);
  formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');
  formatted = formatted.replace(/\n\n/g, '<br><br>');
  return formatted;
}
