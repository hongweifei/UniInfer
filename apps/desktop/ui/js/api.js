export class ApiError extends Error {
  constructor(message, code = 'api_error', status = 0) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
    this.status = status;
  }
}

export class ApiClient {
  constructor({ base, token }) {
    this.base = base.replace(/\/$/, '');
    this.token = token;
  }

  headers(extra = {}) {
    return {
      Authorization: `Bearer ${this.token}`,
      ...extra,
    };
  }

  async request(path, { method = 'GET', body, headers = {}, signal } = {}) {
    const isForm = body instanceof FormData;
    const response = await fetch(`${this.base}${path}`, {
      method,
      headers: this.headers(isForm ? headers : { 'Content-Type': 'application/json', ...headers }),
      body: isForm ? body : body == null ? undefined : JSON.stringify(body),
      signal,
    });
    const text = await response.text();
    let payload = {};
    if (text) {
      try { payload = JSON.parse(text); } catch { payload = { message: text }; }
    }
    if (!response.ok) {
      throw new ApiError(
        payload.message ?? payload.error?.message ?? `HTTP ${response.status}`,
        payload.code ?? 'http_error',
        response.status,
      );
    }
    return payload;
  }

  models() { return this.request('/v1/models'); }
  devices() { return this.request('/v1/devices'); }
  config() { return this.request('/v1/config'); }
  saveConfig(config) { return this.request('/v1/config', { method: 'PUT', body: config }); }
  backends() { return this.request('/v1/backends'); }
  removeBackend(id) {
    return this.request(`/v1/backends/${encodeURIComponent(id)}`, { method: 'DELETE' });
  }
  inspectModel(path) {
    return this.request('/v1/models/inspect', { method: 'POST', body: { path } });
  }
  importModel(path) {
    return this.request('/v1/models/import', { method: 'POST', body: { path } });
  }
  loadModel(id) {
    return this.request(`/v1/models/${encodeURIComponent(id)}/load`, { method: 'POST' });
  }
  unloadModel(id) {
    return this.request(`/v1/models/${encodeURIComponent(id)}/unload`, { method: 'POST' });
  }

  async transcribe(file, model) {
    const form = new FormData();
    form.append('model', model);
    form.append('response_format', 'verbose_json');
    form.append('file', file, file.name);
    return this.request('/v1/audio/transcriptions', { method: 'POST', body: form });
  }

  generateImage(payload) {
    return this.request('/v1/images/generations', { method: 'POST', body: payload });
  }

  async streamChat({ model, messages, maxTokens, temperature, enableThinking, signal, onDelta, onReasoning }) {
    const response = await fetch(`${this.base}/v1/chat/completions`, {
      method: 'POST',
      headers: this.headers({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({
        model,
        messages,
        stream: true,
        max_tokens: maxTokens,
        temperature,
        enable_thinking: enableThinking,
      }),
      signal,
    });
    if (!response.ok) {
      const text = await response.text();
      let payload = {};
      try { payload = JSON.parse(text); } catch { payload = { message: text }; }
      throw new ApiError(
        payload.message ?? payload.error?.message ?? `HTTP ${response.status}`,
        payload.code ?? 'http_error',
        response.status,
      );
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let content = '';
    let reasoning = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let index;
      while ((index = buffer.indexOf('\n\n')) >= 0) {
        const frame = buffer.slice(0, index);
        buffer = buffer.slice(index + 2);
        for (const line of frame.split('\n')) {
          if (!line.startsWith('data:')) continue;
          const payload = line.slice(5).trim();
          if (!payload || payload === '[DONE]') continue;
          let event;
          try { event = JSON.parse(payload); } catch { continue; }
          if (event.error) {
            throw new ApiError(event.error.message ?? '生成失败', event.error.code ?? 'stream_error');
          }
          const delta = event.choices?.[0]?.delta;
          const reasoningDelta = delta?.reasoning_content ?? delta?.reasoning;
          if (typeof reasoningDelta === 'string' && reasoningDelta) {
            reasoning += reasoningDelta;
            onReasoning?.(reasoningDelta, reasoning);
          }
          if (typeof delta?.content === 'string' && delta.content) {
            content += delta.content;
            onDelta?.(delta.content, content);
          }
        }
      }
    }
    return content;
  }
}
