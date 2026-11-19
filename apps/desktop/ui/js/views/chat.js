import { h, clear, autoGrow } from '../dom.js';
import { icon } from '../icons.js';
import { appendMessage, ensureConversation } from '../state.js';
import { readAsDataURL, errorMessage, isChatModel, isImageModel, isAudioModel } from '../format.js';
import { toast, emptyState, renderText, openModal } from '../components/ui.js';
import { messageNode } from '../components/chat-message.js';
import { putMedia, hydrateMedia } from '../media-store.js';

const MAX_ATTACHMENTS = 8;

export function mountChat(root, ctx) {
  const { state, api } = ctx;
  const conversation = ensureConversation(state, ctx.currentModel()?.id);
  const attachments = [];
  let controller = null;
  let draftText = state.ui.chatDrafts?.[conversation.id] ?? state.ui.chatDraft ?? '';
  if (draftText && !state.ui.chatDrafts?.[conversation.id]) {
    state.ui.chatDrafts ??= {};
    state.ui.chatDrafts[conversation.id] = draftText;
    state.ui.chatDraft = '';
    ctx.persistSoon?.();
  }

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
  const audioInput = h('input', {
    type: 'file', accept: '.wav,audio/wav,audio/x-wav', class: 'visually-hidden',
    onChange: (event) => { const file = event.target.files?.[0]; if (file) void transcribeAudio(file); audioInput.value = ''; },
  });
  const imageModels = ctx.models().filter((model) => isImageModel(model) && !model.route_error);
  const audioModels = ctx.models().filter((model) => isAudioModel(model) && !model.route_error);
  const textarea = h('textarea', {
    rows: 1,
    value: draftText,
    placeholder: '输入消息，Enter 发送，Shift+Enter 换行',
    attrs: { 'aria-label': '消息输入框' },
  });
  const send = h('button', {
    class: 'btn btn-icon primary', type: 'submit', title: '发送', html: icon('send', 16),
  });
  const stop = h('button', {
    class: 'btn btn-icon ghost', type: 'button', title: '停止', html: icon('stop', 15), onClick: () => controller?.abort(),
  });
  const audioButton = audioModels.length ? h('button', {
    class: 'btn btn-icon ghost', type: 'button', title: '音频转写',
    onClick: () => audioInput.click(), html: icon('wave', 17),
  }) : null;
  const imageButton = imageModels.length ? h('button', {
    class: 'btn btn-icon ghost', type: 'button', title: '生成图片',
    onClick: () => openImageDialog(), html: icon('image', 17),
  }) : null;
  const hint = h('span', { class: 'composer-hint' });
  const modelHint = h('span', { class: 'faint' });
  const modelSelect = h('select', { class: 'composer-model', attrs: { 'aria-label': '对话模型' } });
  for (const model of ctx.models().filter((item) => isChatModel(item) && !item.route_error)) {
    modelSelect.append(h('option', { value: model.id, text: model.id }));
  }
  const conversationModel = ctx.models().find((model) => model.id === conversation.model && isChatModel(model) && !model.route_error);
  if (conversationModel && ctx.currentModel()?.id !== conversationModel.id) {
    ctx.setModel(conversationModel.id);
  }
  if (ctx.currentModel()) modelSelect.value = ctx.currentModel().id;
  modelSelect.addEventListener('change', () => {
    ctx.setModel(modelSelect.value);
    conversation.model = modelSelect.value;
    ctx.persistSoon?.();
  });
  const thinkingToggle = h('button', {
    class: 'btn sm ghost composer-thinking',
    type: 'button',
    title: '切换深度思考',
    attrs: { 'aria-pressed': String(Boolean(state.chatOptions.thinking)) },
    onClick: () => {
      state.chatOptions.thinking = !state.chatOptions.thinking;
      ctx.persist();
      updateThinkingToggle();
    },
  }, h('span', { html: icon('sparkles', 13) }), h('span', { text: '深度思考' }));
  const form = h('form', { class: 'composer', onSubmit: (event) => { event.preventDefault(); void submit(); } },
    attachmentRow,
    h('div', { class: 'composer-row' },
      h('div', { class: 'composer-tools' },
        h('button', { class: 'btn btn-icon ghost', type: 'button', title: '添加图片', onClick: () => fileInput.click(), html: icon('paperclip', 17) }),
        audioButton,
        imageButton,
      ),
      textarea,
      h('div', { class: 'composer-tools' }, stop, send),
    ),
    h('div', { class: 'composer-meta' }, modelSelect, thinkingToggle, modelHint, h('span', { class: 'spacer' }), hint),
    fileInput,
    audioInput,
  );
  const composerWrap = h('div', { class: 'composer-wrap' }, form);
  root.append(h('div', { class: 'chat-view' }, scroller, composerWrap));
  const grow = autoGrow(textarea);

  async function transcribeAudio(file) {
    if (!audioModels.length) {
      toast('没有可用的语音转写模型', 'error');
      return;
    }
    if (!/audio\/|wav/i.test(file.type || file.name) || !file.name.toLowerCase().endsWith('.wav')) {
      toast('语音输入目前支持 WAV', 'error');
      return;
    }
    if (file.size > 128 * 1024 * 1024) {
      toast('音频文件超过 128 MiB', 'error');
      return;
    }
    const button = audioButton;
    if (button) button.disabled = true;
    hint.textContent = '正在转写音频…';
    try {
      const result = await ctx.api.transcribe(file, audioModels[0].id);
      const transcript = String(result.text ?? '').trim();
      if (!transcript) throw new Error('转写结果为空');
      textarea.value = textarea.value.trim()
        ? `${textarea.value.trimEnd()}\n${transcript}`
        : transcript;
      state.ui.chatDrafts[conversation.id] = textarea.value;
      state.ui.chatDraft = '';
      ctx.persistSoon?.();
      grow();
      updateHint();
      toast('音频已转写为文字', 'success');
    } catch (error) {
      toast(errorMessage(error), 'error');
      updateHint();
    } finally {
      if (button) button.disabled = false;
    }
  }

  function openImageDialog() {
    if (!imageModels.length) {
      toast('没有可用的图像生成模型', 'error');
      return;
    }
    const prompt = h('textarea', {
      class: 'textarea', rows: 3,
      placeholder: '描述要生成的图片',
      value: textarea.value.trim(),
    });
    const model = h('select', { class: 'select' },
      ...imageModels.map((item) => h('option', { value: item.id, text: item.id })),
    );
    const size = h('select', { class: 'select' },
      ['512x512', '768x768', '1024x1024'].map((value) => h('option', { value, text: value })),
    );
    const steps = h('input', { class: 'input', type: 'number', min: '1', max: '100', value: '20' });
    const guidance = h('input', { class: 'input', type: 'number', min: '0', max: '30', step: '0.5', value: '7' });
    let generating = false;
    openModal({
      title: '生成图片',
      body: h('div', { class: 'tool-modal' },
        h('div', { class: 'field' }, h('label', { text: '模型' }), model),
        h('div', { class: 'field' }, h('label', { text: '提示词' }), prompt),
        h('div', { class: 'param-grid' },
          h('div', { class: 'field' }, h('label', { text: '尺寸' }), size),
          h('div', { class: 'field' }, h('label', { text: '步数' }), steps),
          h('div', { class: 'field' }, h('label', { text: '引导强度' }), guidance),
        ),
      ),
      actions: [
        { label: '取消', kind: 'ghost' },
        {
          label: '生成图片', kind: 'primary',
          onClick: async () => {
            if (generating) return false;
            const text = prompt.value.trim();
            if (!text) { toast('请先输入提示词', 'error'); return false; }
            generating = true;
            try {
              const response = await ctx.api.generateImage({
                model: model.value,
                prompt: text,
                n: 1,
                size: size.value,
                response_format: 'b64_json',
                steps: Number(steps.value) || 1,
                guidance: Number(guidance.value) || 0,
              });
              const image = response.data?.[0];
              if (!image?.b64_json) throw new Error('响应中没有图片');
              const url = `data:${image.mime_type ?? 'image/png'};base64,${image.b64_json}`;
              const generated = appendMessage(state, conversation, {
                role: 'assistant',
                content: `生成图片：${text}`,
                images: [url],
                generated: true,
              });
              try {
                await putMedia(generated.id, url);
                generated.mediaKey = generated.id;
                ctx.persist();
              } catch {
                toast('图片已生成，但本地缓存失败', 'error');
              }
              renderHistory();
              scroller.scrollTop = scroller.scrollHeight;
              toast('图片已生成', 'success');
              return true;
            } catch (error) {
              toast(errorMessage(error), 'error');
              return false;
            } finally {
              generating = false;
            }
          },
        },
      ],
    });
  }

  function updateThinkingToggle() {
    const enabled = Boolean(state.chatOptions.thinking);
    thinkingToggle.classList.toggle('ghost', !enabled);
    thinkingToggle.classList.toggle('active', enabled);
    thinkingToggle.setAttribute('aria-pressed', String(enabled));
    thinkingToggle.title = enabled ? '深度思考已开启，点击关闭' : '深度思考已关闭，点击开启';
  }

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

  async function hydrateConversationMedia() {
    if (await hydrateMedia(conversation.messages)) renderHistory();
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
    delete state.ui.chatDrafts[conversation.id];
    state.ui.chatDraft = '';
    ctx.persistSoon?.();
    grow();
    renderHistory();
    scroller.scrollTop = scroller.scrollHeight;

    const assistant = appendMessage(state, conversation, {
      role: 'assistant', content: '', pending: true,
    });
    const { node, text: textNode, reasoning, reasoningBody, reasoningSummary } = messageNode(assistant);
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
      let reasoningFull = '';
      let contentStarted = false;
      let userToggledReasoning = false;
      reasoning?.querySelector('summary')?.addEventListener('click', () => {
        userToggledReasoning = true;
      });
      await api.streamChat({
        model: model.id,
        messages: history,
        maxTokens: state.chatOptions.maxTokens,
        temperature: state.chatOptions.temperature,
        enableThinking: state.chatOptions.thinking,
        signal: abort.signal,
        onReasoning: (delta) => {
          reasoningFull += delta;
          assistant.reasoning = reasoningFull;
          if (reasoning) {
            reasoning.hidden = false;
            if (!contentStarted) reasoning.open = true;
            reasoningSummary.textContent = '正在思考…';
            reasoningBody.textContent = reasoningFull;
          }
          scroller.scrollTop = scroller.scrollHeight;
        },
        onDelta: (delta) => {
          if (!contentStarted) {
            contentStarted = true;
            reasoningSummary.textContent = '思考过程';
            if (reasoning && reasoning.open && !userToggledReasoning) reasoning.open = false;
          }
          full += delta;
          assistant.content = full;
          textNode.textContent = full;
          scroller.scrollTop = scroller.scrollHeight;
        },
      });
      assistant.pending = false;
      assistant.content = full || '（空响应）';
      renderText(textNode, assistant.content);
      if (assistant.reasoning) {
        reasoningSummary.textContent = '思考过程';
        renderText(reasoningBody, assistant.reasoning);
      }
      node.replaceWith(messageNode(assistant).node);
      conversation.updatedAt = Date.now();
    } catch (error) {
      if (error.name === 'AbortError') {
        assistant.pending = false;
        assistant.content = `${assistant.content}\n\n[已停止]`;
        renderText(textNode, assistant.content);
        node.replaceWith(messageNode(assistant).node);
      } else {
        assistant.pending = false;
        assistant.error = true;
        assistant.content = errorMessage(error);
        node.classList.add('error');
        renderText(textNode, assistant.content);
        node.replaceWith(messageNode(assistant).node);
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

  textarea.addEventListener('input', () => {
    if (textarea.value) state.ui.chatDrafts[conversation.id] = textarea.value;
    else delete state.ui.chatDrafts[conversation.id];
    state.ui.chatDraft = '';
    updateHint();
    ctx.persistSoon?.();
  });
  textarea.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });
  scroller.addEventListener('dragover', (event) => event.preventDefault());
  scroller.addEventListener('drop', (event) => {
    event.preventDefault();
    const files = [...event.dataTransfer.files];
    const audio = files.find((file) => file.name.toLowerCase().endsWith('.wav') || file.type.startsWith('audio/'));
    if (audio) void transcribeAudio(audio);
    const images = files.filter((file) => file.type.startsWith('image/'));
    if (images.length) void addFiles(images);
  });
  send.disabled = true;
  stop.disabled = true;
  renderAttachments();
  updateThinkingToggle();
  renderHistory();
  updateHint();
  grow();
  void hydrateConversationMedia();
  requestAnimationFrame(() => {
    grow();
    textarea.focus();
    textarea.setSelectionRange(textarea.value.length, textarea.value.length);
  });

  const onModel = () => updateHint();
  document.addEventListener('uniinfer:model', onModel);
  return () => {
    document.removeEventListener('uniinfer:model', onModel);
    controller?.abort();
  };
}
