import { uid, now } from './format.js';
import { clampText } from './dom.js';

const STORAGE_KEY = 'uniinfer.workspace.v1';
const defaultOptions = { maxTokens: 512, temperature: 0.7 };

function freshState() {
  return {
    view: 'chat',
    theme: 'system',
    selectedModel: null,
    conversations: [],
    activeId: null,
    chatOptions: { ...defaultOptions },
  };
}

export function loadState() {
  const base = freshState();
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return base;
    const saved = JSON.parse(raw);
    return {
      ...base,
      ...saved,
      chatOptions: { ...defaultOptions, ...(saved.chatOptions ?? {}) },
      conversations: Array.isArray(saved.conversations) ? saved.conversations : [],
    };
  } catch {
    return base;
  }
}

export function saveState(state) {
  try {
    const compact = {
      ...state,
      conversations: state.conversations.slice(0, 40).map((conversation) => ({
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

export function recentConversations(state, limit = 8) {
  return [...state.conversations]
    .sort((a, b) => b.updatedAt - a.updatedAt)
    .slice(0, limit);
}
