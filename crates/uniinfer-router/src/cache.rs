//! LRU cache of loaded text-model slots.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use uniinfer_core::{EngineError, Result};
use uniinfer_service::ModelSlot;

pub(crate) struct ModelCache {
    slots: Arc<RwLock<HashMap<String, Arc<ModelSlot>>>>,
    lru: Arc<Mutex<VecDeque<String>>>,
    max_loaded: Arc<AtomicUsize>,
}

impl ModelCache {
    pub(crate) fn new(slots: HashMap<String, Arc<ModelSlot>>, max_loaded: usize) -> Self {
        let lru = slots.keys().cloned().collect::<VecDeque<_>>();
        Self {
            slots: Arc::new(RwLock::new(slots)),
            lru: Arc::new(Mutex::new(lru)),
            max_loaded: Arc::new(AtomicUsize::new(max_loaded)),
        }
    }

    pub(crate) fn all(&self) -> Vec<Arc<ModelSlot>> {
        self.slots
            .read()
            .map(|slots| slots.values().cloned().collect())
            .unwrap_or_default()
    }

    pub(crate) fn get(&self, id: &str) -> Option<Arc<ModelSlot>> {
        let slot = self
            .slots
            .read()
            .ok()
            .and_then(|slots| slots.get(id).cloned())?;
        self.touch(id);
        Some(slot)
    }

    pub(crate) fn peek(&self, id: &str) -> Option<Arc<ModelSlot>> {
        self.slots
            .read()
            .ok()
            .and_then(|slots| slots.get(id).cloned())
    }

    pub(crate) fn insert(&self, id: &str, slot: Arc<ModelSlot>) -> Result<()> {
        {
            let mut slots = self
                .slots
                .write()
                .map_err(|_| EngineError::Other("model catalog lock poisoned".into()))?;
            slots.entry(id.to_owned()).or_insert(slot);
        }
        self.touch(id);
        self.evict();
        Ok(())
    }

    pub(crate) fn remove(&self, id: &str) -> bool {
        let removed = self
            .slots
            .write()
            .map(|mut slots| slots.remove(id).is_some())
            .unwrap_or(false);
        if let Ok(mut lru) = self.lru.lock() {
            lru.retain(|entry| entry != id);
        }
        removed
    }

    /// Evict the least-recently-used model other than `except`.
    pub(crate) fn evict_lru_except(&self, except: Option<&str>) -> Option<String> {
        let mut lru = self.lru.lock().ok()?;
        loop {
            let id = lru.pop_front()?;
            if except.is_some_and(|except| except == id) {
                continue;
            }
            let removed = self
                .slots
                .write()
                .map(|mut slots| slots.remove(&id).is_some())
                .unwrap_or(false);
            if removed {
                return Some(id);
            }
        }
    }

    pub(crate) fn set_max_loaded(&self, max_loaded: usize) {
        self.max_loaded.store(max_loaded, Ordering::SeqCst);
        self.evict();
    }

    fn touch(&self, id: &str) {
        if let Ok(mut lru) = self.lru.lock() {
            lru.retain(|entry| entry != id);
            lru.push_back(id.to_owned());
        }
    }

    fn evict(&self) {
        let max_loaded = self.max_loaded.load(Ordering::SeqCst);
        if max_loaded == 0 {
            return;
        }
        let Ok(mut lru) = self.lru.lock() else {
            return;
        };
        while lru.len() > max_loaded {
            let Some(oldest) = lru.pop_front() else {
                break;
            };
            if let Ok(mut slots) = self.slots.write() {
                slots.remove(&oldest);
            }
        }
    }
}
