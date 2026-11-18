function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function formatInlineMarkdown(escapedLine) {
  return escapedLine
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
}

function renderMessageMarkup(text) {
  const safe = escapeHtml(text || '').replace(/\r\n/g, '\n');
  const lines = safe.split('\n');
  const blocks = [];
  let listType = null;
  let listItems = [];

  function flushList() {
    if (listItems.length > 0 && listType) {
      const tag = listType === 'ol' ? 'ol' : 'ul';
      blocks.push(`<${tag}>${listItems.map((item) => `<li>${item}</li>`).join('')}</${tag}>`);
      listItems = [];
      listType = null;
    }
  }

  for (const rawLine of lines) {
    const trimmed = rawLine.trim();
    if (!trimmed) {
      flushList();
      continue;
    }

    const bulletMatch = trimmed.match(/^[-*]\s+(.+)$/);
    const orderedMatch = trimmed.match(/^\d+\.\s+(.+)$/);

    if (bulletMatch) {
      if (listType && listType !== 'ul') flushList();
      listType = 'ul';
      listItems.push(formatInlineMarkdown(bulletMatch[1]));
    } else if (orderedMatch) {
      if (listType && listType !== 'ol') flushList();
      listType = 'ol';
      listItems.push(formatInlineMarkdown(orderedMatch[1]));
    } else {
      flushList();
      blocks.push(`<p>${formatInlineMarkdown(trimmed)}</p>`);
    }
  }

  flushList();
  return blocks.join('') || '<p></p>';
}

const AI_SPARK_SVG = `
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <path
      d="M12 3a9 9 0 1 0 9 9v-1.5h-7.5v3h4.2A5.5 5.5 0 1 1 12 6.5c1.66 0 3.15.74 4.16 1.9l2.44-2.44A8.96 8.96 0 0 0 12 3Z"
      fill="currentColor"
    />
  </svg>
`;

const USER_AVATAR_SVG = `
  <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
    <path d="M12 12c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm0 2c-3.33 0-10 1.67-10 5v2h20v-2c0-3.33-6.67-5-10-5z"/>
  </svg>
`;

export class ChatController {
  constructor({ messagesEl, formEl, inputEl, sendBtn, clearBtn, getVisualContext }) {
    this.messagesEl = messagesEl;
    this.formEl = formEl;
    this.inputEl = inputEl;
    this.sendBtn = sendBtn;
    this.clearBtn = clearBtn;
    this.getVisualContext = getVisualContext;

    this.history = [];
    this.isStreaming = false;

    this.renderWelcomeMessage();
    this.bindEvents();
  }

  bindEvents() {
    if (this.formEl) {
      this.formEl.addEventListener('submit', (e) => {
        e.preventDefault();
        const text = (this.inputEl?.value || '').trim();
        if (!text || this.isStreaming) return;
        this.inputEl.value = '';
        this.sendMessage(text);
      });
    }

    if (this.clearBtn) {
      this.clearBtn.addEventListener('click', () => {
        if (this.isStreaming) return;
        this.history = [];
        this.messagesEl.innerHTML = '';
        this.renderWelcomeMessage();
      });
    }
  }

  renderWelcomeMessage() {
    const { row } = this.createMessageRow(
      'assistant',
      'Ask me anything about the objects detected in your camera or uploaded picture, or any general knowledge question.'
    );
    this.messagesEl.appendChild(row);
  }

  createMessageRow(role, text) {
    const row = document.createElement('div');
    const bubble = document.createElement('div');

    if (role === 'user') {
      row.className = 'chat-row chat-row-user';
      bubble.className = 'chat-bubble chat-bubble-user';
      bubble.textContent = text;

      const avatar = document.createElement('div');
      avatar.className = 'chat-avatar-user';
      avatar.innerHTML = USER_AVATAR_SVG;

      row.appendChild(bubble);
      row.appendChild(avatar);
    } else if (role === 'error') {
      row.className = 'chat-row chat-row-assistant';
      const avatar = document.createElement('div');
      avatar.className = 'chat-avatar-ai';
      avatar.innerHTML = AI_SPARK_SVG;

      bubble.className = 'chat-bubble chat-bubble-error';
      bubble.textContent = text;

      row.appendChild(avatar);
      row.appendChild(bubble);
    } else {
      row.className = 'chat-row chat-row-assistant';
      const avatar = document.createElement('div');
      avatar.className = 'chat-avatar-ai';
      avatar.innerHTML = AI_SPARK_SVG;

      bubble.className = 'chat-bubble chat-bubble-assistant';
      bubble.innerHTML = renderMessageMarkup(text);

      row.appendChild(avatar);
      row.appendChild(bubble);
    }

    return { row, bubble };
  }

  createTypingRow() {
    const row = document.createElement('div');
    row.className = 'chat-row chat-row-assistant';

    const avatar = document.createElement('div');
    avatar.className = 'chat-avatar-ai';
    avatar.innerHTML = AI_SPARK_SVG;

    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble chat-bubble-assistant';
    bubble.innerHTML = '<span class="typing-dots" aria-label="AI is typing"><span></span><span></span><span></span></span>';

    row.appendChild(avatar);
    row.appendChild(bubble);
    return { row, bubble };
  }

  scrollToBottom() {
    if (this.messagesEl) {
      this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
    }
  }

  setBusy(busy) {
    this.isStreaming = busy;
    if (this.sendBtn) this.sendBtn.disabled = busy;
    if (this.inputEl) this.inputEl.disabled = busy;
  }

  renderCodeExportBubble() {
    const row = document.createElement('div');
    row.className = 'chat-row chat-row-assistant';

    const avatar = document.createElement('div');
    avatar.className = 'chat-avatar-ai';
    avatar.innerHTML = AI_SPARK_SVG;

    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble chat-bubble-assistant';
    bubble.innerHTML = `
      <p><strong>Working Project Folder Ready</strong></p>
      <p>Download the complete source folder (HTML, CSS, vanilla JS, Express server, and package config — without private API keys):</p>
      <button type="button" class="chat-download-btn">
        Download Project Folder (.zip)
      </button>
      <p class="chat-export-note">Includes all 14 flat files (<code>index.html</code>, <code>server.mjs</code>, <code>package.json</code>, <code>app.js</code>, <code>detector.js</code>, <code>chat.js</code>, <code>theme.js</code>, <code>index.css</code>, etc.) — no subfolders needed.</p>
      <p class="chat-export-note">To run the chatbot on your own host, deploy with Node.js (<code>npm install && npm run build && npm start</code>) and set your own <code>GROQ_API_KEY</code> (or <code>GEMINI_API_KEY</code>) in <code>.env</code>.</p>
    `;

    const downloadBtn = bubble.querySelector('.chat-download-btn');
    downloadBtn?.addEventListener('click', async () => {
      downloadBtn.disabled = true;
      downloadBtn.textContent = 'Preparing ZIP...';
      try {
        const res = await fetch('/api/export-project?code=BRUHHD2DCODEFILES');
        if (!res.ok) throw new Error('Download failed.');
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'ai-vision-project.zip';
        document.body.appendChild(link);
        link.click();
        link.remove();
        setTimeout(() => URL.revokeObjectURL(url), 5000);
        downloadBtn.textContent = 'Download Project Folder (.zip)';
      } catch {
        downloadBtn.textContent = 'Download Failed — Try Again';
      } finally {
        downloadBtn.disabled = false;
      }
    });

    row.appendChild(avatar);
    row.appendChild(bubble);
    return row;
  }

  async sendMessage(userText) {
    if (!userText || this.isStreaming) return;

    if (userText.trim().toUpperCase() === '#BRUHHD2DCODEFILES') {
      const { row: userRow } = this.createMessageRow('user', userText);
      this.messagesEl.appendChild(userRow);
      const exportRow = this.renderCodeExportBubble();
      this.messagesEl.appendChild(exportRow);
      this.scrollToBottom();
      this.inputEl?.focus();
      return;
    }

    this.setBusy(true);

    const { row: userRow } = this.createMessageRow('user', userText);
    this.messagesEl.appendChild(userRow);
    this.history.push({ role: 'user', text: userText });

    const { row: aiRow, bubble: assistantBubble } = this.createTypingRow();
    this.messagesEl.appendChild(aiRow);
    this.scrollToBottom();

    const context = typeof this.getVisualContext === 'function' ? this.getVisualContext() : {};
    let fullResponse = '';
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 55000);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: this.history,
          detectedObjects: context.detectedObjects || [],
          imageDataUrl: context.imageDataUrl || null,
          mediaMode: context.mediaMode || 'idle',
        }),
        signal: controller.signal,
      });

      const contentType = response.headers.get('content-type') || '';
      if (!response.ok || contentType.includes('application/json')) {
        const errPayload = await response.json().catch(() => ({}));
        throw new Error(errPayload.error || `Request failed with status ${response.status}`);
      }

      if (!response.body) {
        throw new Error('Streaming response body unavailable.');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split('\n\n');
        buffer = events.pop() || '';

        for (const rawEvent of events) {
          const lines = rawEvent.split('\n');
          let eventType = 'message';
          let dataLine = '';

          for (const line of lines) {
            if (line.startsWith('event:')) {
              eventType = line.slice(6).trim();
            } else if (line.startsWith('data:')) {
              dataLine += line.slice(5).trim();
            }
          }

          if (!dataLine) continue;
          let parsed;
          try {
            parsed = JSON.parse(dataLine);
          } catch {
            continue;
          }

          if (eventType === 'delta' && parsed.text) {
            fullResponse += parsed.text;
            assistantBubble.innerHTML = renderMessageMarkup(fullResponse);
            this.scrollToBottom();
          } else if (eventType === 'error') {
            throw new Error(parsed.message || 'AI response error.');
          }
        }
      }

      if (fullResponse.trim()) {
        this.history.push({ role: 'model', text: fullResponse });
      } else {
        assistantBubble.innerHTML = renderMessageMarkup('No response received.');
      }
    } catch (error) {
      assistantBubble.className = 'chat-bubble chat-bubble-error';
      if (error?.name === 'AbortError') {
        assistantBubble.textContent = 'Request timed out. Please try again.';
      } else {
        assistantBubble.textContent = error instanceof Error ? error.message : 'Unable to connect to AI Chatbot.';
      }
    } finally {
      clearTimeout(timeoutId);
      this.setBusy(false);
      this.scrollToBottom();
      this.inputEl?.focus();
    }
  }
}
