import { h, clear, autoGrow } from '../dom.js';
import { icon } from '../icons.js';
import { appendMessage, ensureConversation } from '../state.js';
import { readAsDataURL, errorMessage, timeAgo, isChatModel } from '../format.js';
import { toast, emptyState, renderText, copyText } from '../components/ui.js';

const MAX_ATTACHMENTS = 8;

function messageNode(message) {
  const isUser = message.role === 'user';
  const text = h('div', { class: 'message-text' });
  renderText(text, message.content);
  const meta = h('div', { class: 'message-meta' },
    !isUser ? h('span', { text: 'UniInfer' }) : null,
    !message.pending ? h('span', { text: timeAgo(message.createdAt ?? Date.now()) }) : null,
  );
  const actions = h('div', { class: 'message-actions' });
  if (!message.pending) {
    actions.append(h('button', {
      class: 'btn btn-icon ghost', type: 'button', title: '复制',
      onClick: () => copyText(message.content ?? ''),
      html: icon('copy', 13),
    }));
  }
  const body = h('div', { class: 'message-body' }, meta, text, actions);
  const node = h('div', { class: `message ${isUser ? 'user' : 'assistant'}${message.error ? ' error' : ''}` },
    h('div', { class: 'avatar', text: isUser ? '你' : 'U' }),
    body,
  );
  if (message.images?.length) {
    const gallery = h('div', { class: 'attachments' });
    for (const url of message.images) {
      gallery.append(h('div', { class: 'attachment' }, h('img', { src: url, alt: '图片附件' })));
    }
    body.insertBefore(gallery, text);
  } else if (message.imageCount) {
    body.insertBefore(h('div', { class: 'row' }, h('span', { class: 'chip', text: `${message.imageCount} 张图片` })), text);
  }
  return { node, text };
}

export function mountChat(root, ctx) {
  const { state, api } = ctx;
  const conversation = ensureConversation(state, ctx.currentModel()?.id);
  const attachments = [];
  let controller = null;

  const scroller = h('div', { class: 'chat-scroll' });
  const thread = h('div', { class: 'chat-content' });
  const hero = h('div', { class: 'chat-empty' });
  const messages = h('div', { class: 'chat-thread', attrs: { 'aria-live': 'polite' } });
  thread.append(messages);
  scroller.append(hero, thread);

  const attachmentRow = h('div', { class: 'attachments' });
  const fileInput = h('input', {
    type: 'file', accept: 'image/*', multiple: true, class: 'visually-hidden',
    onChange: (event) => addFiles([...event.target.files]),
  });
  const textarea = h('textarea', {
    rows: 1,
    placeholder: '输入消息，Enter 发送，Shift+Enter 换行',
    attrs: { 'aria-label': '消息输入框' },
  });
  const send = h('button', {
    class: 'btn btn-icon primary', type: 'submit', title: '发送', html: icon('send', 16),
  });
  const stop = h('button', {
    class: 'btn btn-icon ghost', type: 'button', title: '停止', html: icon('stop', 15), onClick: () => controller?.abort(),
  });
  const hint = h('span', { class: 'composer-hint' });
  const modelHint = h('span', { class: 'faint' });
  const modelSelect = h('select', { class: 'composer-model', attrs: { 'aria-label': '对话模型' } });
  for (const model of ctx.models().filter((item) => isChatModel(item) && !item.route_error)) {
    modelSelect.append(h('option', { value: model.id, text: model.id }));
  }
  if (ctx.currentModel()) modelSelect.value = ctx.currentModel().id;
  modelSelect.addEventListener('change', () => ctx.setModel(modelSelect.value));
  const form = h('form', { class: 'composer', onSubmit: (event) => { event.preventDefault(); void submit(); } },
    attachmentRow,
    h('div', { class: 'composer-row' },
      h('div', { class: 'composer-tools' },
        h('button', { class: 'btn btn-icon ghost', type: 'button', title: '添加图片', onClick: () => fileInput.click(), html: icon('paperclip', 17) }),
      ),
      textarea,
      h('div', { class: 'composer-tools' }, stop, send),
    ),
    h('div', { class: 'composer-meta' }, modelSelect, modelHint, h('span', { class: 'spacer' }), hint),
    fileInput,
  );
  const composerWrap = h('div', { class: 'composer-wrap' }, form);
  root.append(h('div', { class: 'chat-view' }, scroller, composerWrap));

  function renderHero() {
    clear(hero);
    const hasMessages = conversation.messages.length > 0;
    hero.style.display = hasMessages ? 'none' : 'grid';
    if (hasMessages) return;
    const suggestions = [
      ['总结内容', '把一段长文压缩成要点', '请总结下面的内容，并列出三条要点：\n'],
      ['识别图片', '读取截图或照片里的信息', null],
      ['比较图片', '分析两张图片的差异', null],
      ['解释代码', '说明代码在做什么', '请解释这段代码的作用：\n```\n\n```'],
    ];
    const cards = suggestions.map(([title, desc, prompt]) => h('button', {
      class: 'suggestion', type: 'button',
      onClick: () => {
        if (prompt === null) { fileInput.click(); return; }
        textarea.value = prompt;
        textarea.focus();
        textarea.setSelectionRange(prompt.length, prompt.length);
      },
    }, h('strong', { text: title }), h('span', { text: desc })));
    hero.append(h('div', { class: 'hero' },
      h('div', { class: 'hero-eyebrow', text: 'LOCAL INFERENCE WORKSPACE' }),
      h('h2', { text: '本地模型，也能像在线助手一样好用。' }),
      h('p', { text: '对话、看图、转写和生成都在本机完成。选择模型后直接输入，或从下面开始。' }),
      h('div', { class: 'suggestion-grid' }, cards),
    ));
  }

  function renderHistory() {
    clear(messages);
    for (const message of conversation.messages) {
      const { node } = messageNode(message);
      messages.append(node);
    }
    const hasMessages = conversation.messages.length > 0;
    renderHero();
    scroller.classList.toggle('is-empty', !hasMessages);
    requestAnimationFrame(() => {
      scroller.scrollTop = hasMessages ? scroller.scrollHeight : 0;
    });
  }

  function updateHint() {
    const model = ctx.currentModel();
    modelHint.textContent = '';
    if (model) modelSelect.value = model.id;
    hint.textContent = model
      ? `Enter 发送 · ${model.loaded ? '已加载' : '按需加载'}`
      : '先选择一个模型';
    const canImage = model?.inputs?.includes('image');
    if (attachments.length && !canImage) {
      hint.textContent = '当前模型不支持图片输入';
    }
    send.disabled = Boolean(controller) || (!textarea.value.trim() && attachments.length === 0);
  }

  function renderAttachments() {
    clear(attachmentRow);
    attachments.forEach((item, index) => {
      attachmentRow.append(h('div', { class: 'attachment' },
        h('img', { src: item.url, alt: item.file.name }),
        h('button', {
          type: 'button', title: '移除', html: icon('x', 11),
          onClick: () => { attachments.splice(index, 1); renderAttachments(); updateHint(); },
        }),
      ));
    });
    attachmentRow.style.display = attachments.length ? 'flex' : 'none';
  }

  async function addFiles(files) {
    const imageFiles = files.filter((file) => file.type.startsWith('image/'));
    if (!imageFiles.length) return;
    if (!ctx.currentModel()?.inputs?.includes('image')) {
      toast('当前模型不支持图片输入，先在模型页选择视觉模型', 'error');
      return;
    }
    for (const file of imageFiles.slice(0, MAX_ATTACHMENTS - attachments.length)) {
      if (file.size > 20 * 1024 * 1024) {
        toast(`${file.name} 超过 20 MiB`, 'error');
        continue;
      }
      attachments.push({ file, url: await readAsDataURL(file) });
    }
    fileInput.value = '';
    renderAttachments();
    updateHint();
  }

  function setStreaming(value) {
    controller = value;
    send.disabled = value || (!textarea.value.trim() && attachments.length === 0);
    stop.disabled = !value;
    textarea.setAttribute('aria-busy', String(value));
    hint.textContent = value ? '生成中…' : `Enter 发送 · ${ctx.currentModel()?.loaded ? '已加载' : '按需加载'}`;
  }

  async function submit() {
    if (controller) return;
    const text = textarea.value.trim();
    if (!text && attachments.length === 0) return;
    const model = ctx.currentModel();
    if (!model) { toast('没有可用模型', 'error'); return; }
    if (attachments.length && !model.inputs?.includes('image')) {
      toast('当前模型不支持图片输入', 'error');
      return;
    }
    const userImages = attachments.map((item) => item.url);
    const userMessage = appendMessage(state, conversation, {
      role: 'user',
      content: text || '请描述这些图片。',
      images: userImages,
      imageCount: userImages.length,
    });
    attachments.length = 0;
    renderAttachments();
    textarea.value = '';
    autoGrow(textarea);
    renderHistory();
    scroller.scrollTop = scroller.scrollHeight;

    const assistant = appendMessage(state, conversation, {
      role: 'assistant', content: '', pending: true,
    });
    const { node, text: textNode } = messageNode(assistant);
    messages.append(node);
    scroller.scrollTop = scroller.scrollHeight;
    const abort = new AbortController();
    setStreaming(abort);
    try {
      const history = conversation.messages
        .filter((item) => item.id !== assistant.id && !item.pending)
        .map((item) => ({ role: item.role, content: item.content }));
      const last = history[history.length - 1];
      if (last && last.role === 'user') {
        if (userImages.length) {
          last.content = [
            { type: 'text', text: text || '请描述这些图片。' },
            ...userImages.map((url) => ({ type: 'image_url', image_url: { url } })),
          ];
        }
      }
      let full = '';
      await api.streamChat({
        model: model.id,
        messages: history,
        maxTokens: state.chatOptions.maxTokens,
        temperature: state.chatOptions.temperature,
        signal: abort.signal,
        onDelta: (delta) => {
          full += delta;
          assistant.content = full;
          textNode.textContent = full;
          scroller.scrollTop = scroller.scrollHeight;
        },
      });
      assistant.pending = false;
      assistant.content = full || '（空响应）';
      renderText(textNode, assistant.content);
      node.querySelector('.message-meta').replaceWith(messageNode(assistant).node.querySelector('.message-meta'));
      conversation.updatedAt = Date.now();
    } catch (error) {
      if (error.name === 'AbortError') {
        assistant.pending = false;
        assistant.content = `${assistant.content}\n\n[已停止]`;
        renderText(textNode, assistant.content);
        const meta = messageNode(assistant).node.querySelector('.message-meta');
        node.querySelector('.message-meta').replaceWith(meta);
      } else {
        assistant.pending = false;
        assistant.error = true;
        assistant.content = errorMessage(error);
        node.classList.add('error');
        renderText(textNode, assistant.content);
        toast(assistant.content, 'error');
      }
    } finally {
      assistant.pending = false;
      ctx.persist();
      setStreaming(null);
      ctx.afterConversationChange?.();
      textarea.focus();
      scroller.scrollTop = scroller.scrollHeight;
    }
  }

  textarea.addEventListener('input', () => { autoGrow(textarea); updateHint(); });
  textarea.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });
  scroller.addEventListener('dragover', (event) => event.preventDefault());
  scroller.addEventListener('drop', (event) => {
    event.preventDefault();
    void addFiles([...event.dataTransfer.files]);
  });
  send.disabled = true;
  stop.disabled = true;
  renderAttachments();
  renderHistory();
  updateHint();
  requestAnimationFrame(() => textarea.focus());

  const onModel = () => updateHint();
  document.addEventListener('uniinfer:model', onModel);
  return () => document.removeEventListener('uniinfer:model', onModel);
}
