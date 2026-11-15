//! Streaming-safe stop-sequence matching.
//!
//! Stop strings are UTF-8, but matching happens on bytes so a sequence can
//! span arbitrary token/chunk boundaries. Bytes that could still become part
//! of a stop sequence are held back until they are known to be safe.

/// Matches stop sequences across arbitrary byte chunks.
pub struct StopMatcher {
    sequences: Vec<Vec<u8>>,
    pending: Vec<u8>,
    matched: bool,
}

impl StopMatcher {
    /// Build a matcher. Empty sequences are ignored.
    #[must_use]
    pub fn new<S: Into<String>>(sequences: impl IntoIterator<Item = S>) -> Self {
        let sequences = sequences
            .into_iter()
            .map(Into::into)
            .filter(|sequence| !sequence.is_empty())
            .map(String::into_bytes)
            .collect();
        Self {
            sequences,
            pending: Vec::new(),
            matched: false,
        }
    }

    /// True when the matcher cannot stop generation.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }

    /// True once a stop sequence has been seen.
    #[must_use]
    pub fn is_matched(&self) -> bool {
        self.matched
    }

    /// Consume a chunk and emit every byte that is known not to belong to a
    /// stop sequence. Returns `false` once a stop sequence matched.
    pub fn push(&mut self, chunk: &[u8], emit: &mut dyn FnMut(&[u8])) -> bool {
        if self.matched {
            return false;
        }
        if self.sequences.is_empty() {
            emit(chunk);
            return true;
        }

        self.pending.extend_from_slice(chunk);
        if let Some(index) = self.find_match() {
            if index > 0 {
                emit(&self.pending[..index]);
            }
            self.pending.clear();
            self.matched = true;
            return false;
        }

        let safe = self.safe_prefix_len();
        if safe > 0 {
            emit(&self.pending[..safe]);
            self.pending.drain(..safe);
        }
        true
    }

    /// Flush bytes still held back because they are a prefix of a stop
    /// sequence. Does nothing after a stop matched.
    pub fn finish(&mut self, emit: &mut dyn FnMut(&[u8])) {
        if self.matched {
            self.pending.clear();
            return;
        }
        if !self.pending.is_empty() {
            emit(&self.pending);
            self.pending.clear();
        }
    }

    fn find_match(&self) -> Option<usize> {
        self.sequences
            .iter()
            .filter_map(|sequence| find(&self.pending, sequence))
            .min()
    }

    /// Longest suffix that is a prefix of some stop sequence.
    fn safe_prefix_len(&self) -> usize {
        let max = self
            .sequences
            .iter()
            .map(Vec::len)
            .max()
            .unwrap_or(0)
            .saturating_sub(1)
            .min(self.pending.len());
        for keep in (1..=max).rev() {
            let suffix = &self.pending[self.pending.len() - keep..];
            if self
                .sequences
                .iter()
                .any(|sequence| sequence.starts_with(suffix))
            {
                return self.pending.len() - keep;
            }
        }
        self.pending.len()
    }
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(chunks: &[&[u8]], stops: &[&str]) -> (Vec<u8>, bool) {
        let mut matcher = StopMatcher::new(stops.iter().copied());
        let mut out = Vec::new();
        for chunk in chunks {
            let keep_going = matcher.push(chunk, &mut |bytes| out.extend_from_slice(bytes));
            if !keep_going {
                break;
            }
        }
        matcher.finish(&mut |bytes| out.extend_from_slice(bytes));
        (out, matcher.is_matched())
    }

    #[test]
    fn stops_across_chunks_without_emitting_the_marker() {
        let (out, matched) = run(&[b"hel", b"lo ST", b"OP world"], &["STOP"]);
        assert_eq!(out, b"hello ");
        assert!(matched);
    }

    #[test]
    fn holds_back_partial_prefix_until_resolved() {
        let (out, matched) = run(&[b"abcST", b"art"], &["STOP"]);
        assert_eq!(out, b"abcSTart");
        assert!(!matched);
    }

    #[test]
    fn chooses_the_earliest_of_several_stops() {
        let (out, matched) = run(&[b"one two three"], &["three", "two"]);
        assert_eq!(out, b"one ");
        assert!(matched);
    }

    #[test]
    fn no_stop_passes_everything() {
        let (out, matched) = run(&[b"abc", b"def"], &[]);
        assert_eq!(out, b"abcdef");
        assert!(!matched);
    }

    #[test]
    fn empty_stop_sequences_are_ignored() {
        let (out, matched) = run(&[b"abc"], &[""]);
        assert_eq!(out, b"abc");
        assert!(!matched);
    }
}
