const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

let recognition = null;
let listening = false;

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}

function openModal(id) {
  const modal = document.getElementById(id);
  if (!modal) return;
  modal.classList.add('open');
  modal.setAttribute('aria-hidden', 'false');
}

function closeModal(id) {
  const modal = document.getElementById(id);
  if (!modal) return;
  modal.classList.remove('open');
  modal.setAttribute('aria-hidden', 'true');
}

$$('[data-close]').forEach((button) => {
  button.addEventListener('click', () => closeModal(button.dataset.close));
});

$$('.modal-backdrop').forEach((modal) => {
  modal.addEventListener('click', (event) => {
    if (event.target === modal) closeModal(modal.id);
  });
});

document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') {
    $$('.modal-backdrop.open').forEach((modal) => closeModal(modal.id));
  }
});

function scrollMessages() {
  const box = $('#messages');
  if (box) box.scrollTop = box.scrollHeight;
}

function addMessage(role, text) {
  const box = $('#messages');
  if (!box) return;
  $('.welcome')?.remove();

  const wrapper = document.createElement('div');
  wrapper.className = `message ${role}`;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = escapeHtml(text).replace(/\n/g, '<br>');
  wrapper.appendChild(bubble);
  box.appendChild(wrapper);
  scrollMessages();
}

function addFileMessage(filename) {
  const box = $('#messages');
  if (!box) return;
  $('.welcome')?.remove();

  const wrapper = document.createElement('div');
  wrapper.className = 'message user file-message';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = `<div class="file-badge">📄 File uploaded</div><div>${escapeHtml(filename)}</div>`;
  wrapper.appendChild(bubble);
  box.appendChild(wrapper);
  scrollMessages();
}

function addSummaryMessage(filename, summary) {
  const box = $('#messages');
  if (!box) return;
  const wrapper = document.createElement('div');
  wrapper.className = 'message assistant';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = `<div class="summary-heading">Summary</div><div>${escapeHtml(summary).replace(/\n/g, '<br>')}</div>`;
  wrapper.appendChild(bubble);
  box.appendChild(wrapper);
  scrollMessages();
}

async function sendMessage(inputType = 'text') {
  const input = $('#messageInput');
  const question = input.value.trim();
  if (!question) return;

  input.value = '';
  input.style.height = 'auto';
  addMessage('user', question);
  $('#sendBtn').disabled = true;

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: question, input_type: inputType})
    });
    const data = await response.json();
    if (!data.ok) throw new Error(data.error || 'Something went wrong.');
    addMessage('assistant', data.answer);
    loadChatHistory();
  } catch (error) {
    addMessage('assistant', `Error: ${error.message}`);
  } finally {
    $('#sendBtn').disabled = false;
    input.focus();
  }
}

$('#sendBtn').addEventListener('click', () => sendMessage());

$('#messageInput').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
});

$('#messageInput').addEventListener('input', (event) => {
  event.target.style.height = 'auto';
  event.target.style.height = Math.min(event.target.scrollHeight, 120) + 'px';
});

$$('.chips button').forEach((button) => {
  button.addEventListener('click', () => {
    $('#messageInput').value = button.textContent.trim();
    sendMessage();
  });
});

async function createNewChat() {
  await fetch('/api/new-chat', {method: 'POST'});
  window.location.reload();
}

$('#newChat').addEventListener('click', createNewChat);
$('#chatMenu').addEventListener('click', createNewChat);

$('#exportCurrent').addEventListener('click', () => {
  window.location.href = '/api/export/current';
});

async function loadChatHistory() {
  const box = $('#chatHistory');
  if (!box) return;

  try {
    const response = await fetch('/api/history');
    const data = await response.json();

    if (!data.length) {
      box.innerHTML = '<div class="empty-history">Your conversations will appear here.</div>';
      return;
    }

    box.innerHTML = data.map((chat) => `
      <div class="chat-history-item" data-id="${chat.id}" role="button" tabindex="0">
        <span class="chat-dot"></span>
        <span class="chat-title">${escapeHtml(chat.title)}</span>
        <button class="delete-chat" title="Delete chat" data-delete="${chat.id}">×</button>
      </div>
    `).join('');

    box.querySelectorAll('.chat-history-item').forEach((item) => {
      item.addEventListener('click', (event) => {
        if (event.target.closest('.delete-chat')) return;
        openHistory(Number(item.dataset.id));
      });
      item.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          openHistory(Number(item.dataset.id));
        }
      });
    });

    box.querySelectorAll('[data-delete]').forEach((button) => {
      button.addEventListener('click', async (event) => {
        event.stopPropagation();
        if (!confirm('Delete this conversation?')) return;
        await fetch('/api/history/' + button.dataset.delete, {method: 'DELETE'});
        loadChatHistory();
      });
    });
  } catch (error) {
    box.innerHTML = '<div class="empty-history">Could not load chats.</div>';
  }
}

async function openHistory(id) {
  try {
    const response = await fetch('/api/history/' + id);
    const data = await response.json();
    const box = $('#messages');
    box.innerHTML = '';
    data.messages.forEach((message) => addMessage(message.role, message.content));
    loadChatHistory();
    scrollMessages();
  } catch (error) {
    addMessage('assistant', `Error: ${error.message}`);
  }
}

$('#refreshChats').addEventListener('click', loadChatHistory);

async function loadAnalytics() {
  const data = await fetch('/api/analytics').then((response) => response.json());
  $('#aChats').textContent = data.total_chats ?? 0;
  $('#aQuestions').textContent = data.total_questions ?? 0;
  $('#aVoice').textContent = data.voice_questions ?? 0;
  $('#aFiles').textContent = data.files_uploaded ?? 0;
  renderBars($('#analyticsDomains'), data.domains || []);
}

function renderBars(box, items) {
  if (!items.length) {
    box.innerHTML = '<p style="color:var(--muted)">No activity yet.</p>';
    return;
  }
  const max = Math.max(...items.map((item) => item.count), 1);
  box.innerHTML = items.map((item) => `
    <div class="bar-row">
      <div class="bar-label"><span>${escapeHtml(item.domain)}</span><strong>${item.count}</strong></div>
      <div class="bar-track"><div class="bar-fill" style="width:${(item.count / max) * 100}%"></div></div>
    </div>
  `).join('');
}

$('#analyticsBtn').addEventListener('click', async () => {
  openModal('analyticsModal');
  try {
    await loadAnalytics();
  } catch (error) {
    $('#analyticsDomains').innerHTML = '<p style="color:var(--muted)">Analytics are temporarily unavailable.</p>';
  }
});

function startVoice() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    addMessage('assistant', 'Voice input is not supported by this browser. Please use Chrome or Edge.');
    return;
  }

  if (listening) {
    recognition.stop();
    return;
  }

  recognition = new SpeechRecognition();
  recognition.lang = 'en-US';
  recognition.interimResults = false;
  recognition.continuous = false;

  recognition.onstart = () => {
    listening = true;
    $('#micBtn').classList.add('listening');
    $('#fileStatus').textContent = 'Listening...';
  };

  recognition.onresult = (event) => {
    const text = event.results[0][0].transcript;
    $('#messageInput').value = text;
    $('#messageInput').style.height = 'auto';
    $('#fileStatus').textContent = '';
    sendMessage('voice');
  };

  recognition.onerror = () => {
    $('#fileStatus').textContent = 'Voice input failed. Please try again.';
  };

  recognition.onend = () => {
    listening = false;
    $('#micBtn').classList.remove('listening');
    setTimeout(() => { if (!listening) $('#fileStatus').textContent = ''; }, 800);
  };

  recognition.start();
}

$('#micBtn').addEventListener('click', startVoice);

$('#attachBtn').addEventListener('click', () => $('#fileInput').click());

$('#fileInput').addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append('file', file);
  $('#fileStatus').textContent = 'Uploading and reading...';
  $('#attachBtn').disabled = true;

  try {
    const response = await fetch('/api/upload', {method: 'POST', body: formData});
    const data = await response.json();
    if (!data.ok) throw new Error(data.error || 'File upload failed.');

    addFileMessage(data.filename);
    addSummaryMessage(data.filename, data.summary);
    $('#fileStatus').textContent = '✓ File summarized';
    loadChatHistory();
    setTimeout(() => { $('#fileStatus').textContent = ''; }, 2200);
  } catch (error) {
    addMessage('assistant', `File error: ${error.message}`);
    $('#fileStatus').textContent = '';
  } finally {
    $('#attachBtn').disabled = false;
    event.target.value = '';
  }
});

async function loadProfile() {
  const profile = await fetch('/api/profile').then((response) => response.json());
  $('#pName').value = profile.name || '';
  $('#pAge').value = profile.age || '';
  $('#pStudies').value = profile.studies || '';
  $('#pUniversity').value = profile.university || '';
  $('#pInterests').value = Array.isArray(profile.interests) ? profile.interests.join(', ') : '';
}

$('#profileBtn').addEventListener('click', async () => {
  openModal('profileModal');
  try { await loadProfile(); } catch (_) {}
});

$('#saveProfile').addEventListener('click', async () => {
  const interests = $('#pInterests').value.split(',').map((value) => value.trim()).filter(Boolean);
  const payload = {
    name: $('#pName').value.trim(),
    age: $('#pAge').value ? Number($('#pAge').value) : '',
    studies: $('#pStudies').value.trim(),
    university: $('#pUniversity').value.trim(),
    interests
  };

  const response = await fetch('/api/profile', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  const data = await response.json();

  if (data.ok) {
    $('#profileStatus').textContent = 'Profile saved successfully.';
    setTimeout(() => closeModal('profileModal'), 700);
  } else {
    $('#profileStatus').textContent = 'Could not save the profile.';
  }
});

$('#themeBtn').addEventListener('click', () => document.body.classList.toggle('dark'));

loadChatHistory();
scrollMessages();
