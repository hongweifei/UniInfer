// Generated media kept out of localStorage; chat history stores only a key.
const DB_NAME = 'uniinfer-media';
const STORE = 'images';

function openDb() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(STORE)) {
        request.result.createObjectStore(STORE);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('打开图片存储失败'));
  });
}

async function transaction(mode, run) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, mode);
    const store = tx.objectStore(STORE);
    let result;
    try {
      result = run(store);
    } catch (error) {
      reject(error);
      return;
    }
    tx.oncomplete = () => resolve(result);
    tx.onerror = () => reject(tx.error ?? new Error('图片存储失败'));
    tx.onabort = () => reject(tx.error ?? new Error('图片存储已中止'));
  });
}

export function putMedia(key, dataUrl) {
  return transaction('readwrite', (store) => store.put(dataUrl, key));
}

export async function getMedia(key) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const request = db.transaction(STORE, 'readonly').objectStore(STORE).get(key);
    request.onsuccess = () => resolve(request.result ?? null);
    request.onerror = () => reject(request.error ?? new Error('读取图片失败'));
  });
}

export function deleteMedia(key) {
  return transaction('readwrite', (store) => store.delete(key));
}

export async function hydrateMedia(messages) {
  let changed = false;
  await Promise.all(messages.map(async (message) => {
    if (!message.mediaKey || message.images?.length) return;
    try {
      const url = await getMedia(message.mediaKey);
      if (url) {
        message.images = [url];
        changed = true;
      }
    } catch {
      // A missing cache should not block chat history.
    }
  }));
  return changed;
}
