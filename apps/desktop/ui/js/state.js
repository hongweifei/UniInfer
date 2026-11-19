import { uid, now } from './format.js';
import { clampText } from './dom.js';

const STORAGE_KEY = 'uniinfer.workspace.v1';
const defaultOptions = { maxTokens: 512, temperature: 0.7, thinking: true };
export const modelScopes = ['chat', 'transcribe', 'images', 'tts', 'video'];

function freshUi() {
  return {
    chatDraft: '',
    chatDrafts: {},
    models: { filter: 'all', tagFilter: 'all', query: '', formatFilter: 'all', loadFilter: 'all' },
    images: {
      prompt: '',
      negative: '',
      size: '512x512',
      steps: 20,
      guidance: 7,
      seed: '',
      last: null,
    },
    transcribe: { result: null },
  };
}

function freshState() {
  return {
    view: 'chat',
    subview: 'online',
    mediaTab: 'transcribe',
    theme: 'system',
    selectedModel: null,
    modelSelection: Object.fromEntries(modelScopes.map((scope) => [scope, null])),
    conversations: [],
    activeId: null,
    chatOptions: { ...defaultOptions },
    ui: freshUi(),
  };
}

function mergeUi(saved) {
  const base = freshUi();
  const ui = saved?.ui ?? {};
  return {
    ...base,
    ...ui,
    chatDraft: typeof ui.chatDraft === 'string' ? ui.chatDraft : base.chatDraft,
    chatDrafts: ui.chatDrafts && typeof ui.chatDrafts === 'object' ? { ...ui.chatDrafts } : {},
    models: { ...base.models, ...(ui.models ?? {}) },
    images: { ...base.images, ...(ui.images ?? {}) },
    transcribe: { ...base.transcribe, ...(ui.transcribe ?? {}) },
  };
}

export function loadState() {
  const base = freshState();
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return base;
    const saved = JSON.parse(raw);
    const merged = {
      ...base,
      ...saved,
      modelSelection: {
        ...base.modelSelection,
        ...(saved.modelSelection ?? {}),
      },
      chatOptions: { ...defaultOptions, ...(saved.chatOptions ?? {}) },
      ui: mergeUi(saved),
      conversations: Array.isArray(saved.conversations)
        ? saved.conversations.map((conversation) => ({
            ...conversation,
            messages: (conversation.messages ?? []).map((message) =>
              message.pending
                ? {
                    ...message,
                    pending: false,
                    error: true,
                    content: message.content || '（生成被中断）',
                  }
                : message
            ),
          }))
        : [],
    };
    if (merged.view === 'transcribe' || merged.view === 'images') {
      merged.mediaTab = merged.view;
      merged.view = 'media';
    }
    if (merged.subview === 'downloads') merged.subview = 'installed';
    return merged;
  } catch {
    return base;
  }
}

export function saveState(state) {
  try {
    const conversations = state.conversations.slice(0, 40);
    const conversationIds = new Set(conversations.map((conversation) => conversation.id));
    const chatDrafts = Object.fromEntries(
      Object.entries(state.ui?.chatDrafts ?? {})
        .filter(([id, text]) => conversationIds.has(id) && typeof text === 'string' && text.trim())
    );
    const compact = {
      ...state,
      ui: { ...state.ui, chatDrafts },
      conversations: conversations.map((conversation) => ({
        ...conversation,
        messages: conversation.messages.slice(-80).map((message) => ({
          ...message,
          images: undefined,
        })),
      })),
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(compact));
  } catch {
    // Local history is best-effort; generation must keep working if storage is full.
  }
}

export function newConversation(state, model) {
  const conversation = {
    id: uid('chat'),
    title: '新对话',
    model: model ?? null,
    createdAt: now(),
    updatedAt: now(),
    messages: [],
  };
  state.conversations.unshift(conversation);
  state.activeId = conversation.id;
  saveState(state);
  return conversation;
}

export function activeConversation(state) {
  return state.conversations.find((item) => item.id === state.activeId) ?? null;
}

export function ensureConversation(state, model) {
  return activeConversation(state) ?? newConversation(state, model);
}

export function appendMessage(state, conversation, message) {
  const item = {
    id: uid('msg'),
    createdAt: now(),
    ...message,
  };
  conversation.messages.push(item);
  conversation.updatedAt = now();
  if (conversation.title === '新对话' && item.role === 'user' && item.content) {
    conversation.title = clampText(item.content, 42) || '新对话';
  }
  saveState(state);
  return item;
}

export function deleteConversation(state, id) {
  state.conversations = state.conversations.filter((conversation) => conversation.id !== id);
  if (state.activeId === id) state.activeId = null;
  saveState(state);
}

export function recentConversations(state, limit = 8) {
  return [...state.conversations]
    .sort((a, b) => b.updatedAt - a.updatedAt)
    .slice(0, limit);
}
