//! Audio and image model caches with LRU limits.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use uniinfer_service::{AudioSlot, ImageSlot};

#[derive(Clone, Default)]
pub(crate) struct MediaCache {
    audio: Arc<Mutex<HashMap<String, Arc<AudioSlot>>>>,
    image: Arc<Mutex<HashMap<String, Arc<ImageSlot>>>>,
    audio_lru: Arc<Mutex<VecDeque<String>>>,
    image_lru: Arc<Mutex<VecDeque<String>>>,
    evictions: Arc<AtomicU64>,
    max_loaded: Arc<AtomicUsize>,
}

impl MediaCache {
    pub(crate) fn new(max_loaded: usize) -> Self {
        Self {
            max_loaded: Arc::new(AtomicUsize::new(max_loaded)),
            ..Self::default()
        }
    }

    pub(crate) fn audio(&self, id: &str) -> Option<Arc<AudioSlot>> {
        self.audio.lock().ok()?.get(id).cloned()
    }

    pub(crate) fn image(&self, id: &str) -> Option<Arc<ImageSlot>> {
        self.image.lock().ok()?.get(id).cloned()
    }

    pub(crate) fn contains_audio(&self, id: &str) -> bool {
        self.audio.lock().is_ok_and(|slots| slots.contains_key(id))
    }

    pub(crate) fn contains_image(&self, id: &str) -> bool {
        self.image.lock().is_ok_and(|slots| slots.contains_key(id))
    }

    pub(crate) fn counts(&self) -> (usize, usize) {
        (
            self.audio.lock().map_or(0, |slots| slots.len()),
            self.image.lock().map_or(0, |slots| slots.len()),
        )
    }

    pub(crate) fn insert_audio(&self, id: &str, slot: Arc<AudioSlot>) {
        if let Ok(mut slots) = self.audio.lock() {
            slots.insert(id.to_owned(), slot);
        }
        touch(&self.audio_lru, id);
        self.evict_audio();
    }

    pub(crate) fn insert_image(&self, id: &str, slot: Arc<ImageSlot>) {
        if let Ok(mut slots) = self.image.lock() {
            slots.insert(id.to_owned(), slot);
        }
        touch(&self.image_lru, id);
        self.evict_image();
    }

    pub(crate) fn remove(&self, id: &str) {
        if let Ok(mut slots) = self.audio.lock() {
            slots.remove(id);
        }
        if let Ok(mut slots) = self.image.lock() {
            slots.remove(id);
        }
        if let Ok(mut lru) = self.audio_lru.lock() {
            lru.retain(|entry| entry != id);
        }
        if let Ok(mut lru) = self.image_lru.lock() {
            lru.retain(|entry| entry != id);
        }
    }

    pub(crate) fn remove_audio(&self, id: &str) -> bool {
        let removed = self
            .audio
            .lock()
            .map(|mut slots| slots.remove(id).is_some())
            .unwrap_or(false);
        if let Ok(mut lru) = self.audio_lru.lock() {
            lru.retain(|entry| entry != id);
        }
        removed
    }

    pub(crate) fn remove_image(&self, id: &str) -> bool {
        let removed = self
            .image
            .lock()
            .map(|mut slots| slots.remove(id).is_some())
            .unwrap_or(false);
        if let Ok(mut lru) = self.image_lru.lock() {
            lru.retain(|entry| entry != id);
        }
        removed
    }

    /// Evict one medium, never the model being loaded.
    pub(crate) fn evict_one(&self, except: &str) -> bool {
        if let Some(id) = pop_lru_except(&self.audio_lru, except) {
            if let Ok(mut slots) = self.audio.lock() {
                slots.remove(&id);
            }
            self.evictions.fetch_add(1, Ordering::Relaxed);
            return true;
        }
        if let Some(id) = pop_lru_except(&self.image_lru, except) {
            if let Ok(mut slots) = self.image.lock() {
                slots.remove(&id);
            }
            self.evictions.fetch_add(1, Ordering::Relaxed);
            return true;
        }
        false
    }

    pub(crate) fn record_eviction(&self) {
        self.evictions.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn evictions(&self) -> u64 {
        self.evictions.load(Ordering::Relaxed)
    }

    pub(crate) fn set_max_loaded(&self, max_loaded: usize) {
        self.max_loaded.store(max_loaded, Ordering::SeqCst);
        self.evict_audio();
        self.evict_image();
    }

    fn evict_audio(&self) {
        let max_loaded = self.max_loaded.load(Ordering::SeqCst);
        if max_loaded == 0 {
            return;
        }
        let Ok(mut lru) = self.audio_lru.lock() else {
            return;
        };
        while lru.len() > max_loaded {
            let Some(oldest) = lru.pop_front() else {
                break;
            };
            if let Ok(mut slots) = self.audio.lock() {
                slots.remove(&oldest);
            }
        }
    }

    fn evict_image(&self) {
        let max_loaded = self.max_loaded.load(Ordering::SeqCst);
        if max_loaded == 0 {
            return;
        }
        let Ok(mut lru) = self.image_lru.lock() else {
            return;
        };
        while lru.len() > max_loaded {
            let Some(oldest) = lru.pop_front() else {
                break;
            };
            if let Ok(mut slots) = self.image.lock() {
                slots.remove(&oldest);
            }
        }
    }
}

fn touch(lru: &Mutex<VecDeque<String>>, id: &str) {
    if let Ok(mut lru) = lru.lock() {
        lru.retain(|entry| entry != id);
        lru.push_back(id.to_owned());
    }
}

fn pop_lru_except(lru: &Mutex<VecDeque<String>>, except: &str) -> Option<String> {
    let mut lru = lru.lock().ok()?;
    loop {
        let id = lru.pop_front()?;
        if id != except {
            return Some(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn touch_moves_an_entry_to_the_back() {
        let lru = Mutex::new(VecDeque::from(["a".into(), "b".into()]));
        touch(&lru, "a");
        assert_eq!(
            lru.lock().expect("lru").iter().cloned().collect::<Vec<_>>(),
            vec!["b".to_owned(), "a".to_owned()]
        );
    }

    #[test]
    fn eviction_skips_the_model_being_loaded() {
        let lru = Mutex::new(VecDeque::from(["a".into(), "b".into()]));
        assert_eq!(pop_lru_except(&lru, "a").as_deref(), Some("b"));
        assert!(pop_lru_except(&lru, "a").is_none());
    }
}
