/// Incremental UTF-8 decoder for byte-level token streams.
#[derive(Debug, Default)]
pub struct Utf8Decoder {
    pending: Vec<u8>,
}

impl Utf8Decoder {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one byte and return text that is complete so far.
    pub fn push(&mut self, byte: u8) -> Option<String> {
        self.pending.push(byte);
        match std::str::from_utf8(&self.pending) {
            Ok(text) => {
                let out = text.to_owned();
                self.pending.clear();
                (!out.is_empty()).then_some(out)
            }
            Err(error) if error.error_len().is_none() => None,
            Err(_) => {
                let out = String::from_utf8_lossy(&self.pending).into_owned();
                self.pending.clear();
                Some(out)
            }
        }
    }

    /// Flush remaining bytes, replacing any incomplete sequence.
    pub fn finish(&mut self) -> String {
        if self.pending.is_empty() {
            String::new()
        } else {
            let out = String::from_utf8_lossy(&self.pending).into_owned();
            self.pending.clear();
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_ascii_one_byte_at_a_time() {
        let mut decoder = Utf8Decoder::new();
        assert_eq!(decoder.push(b'h'), Some("h".into()));
        assert_eq!(decoder.push(b'i'), Some("i".into()));
        assert_eq!(decoder.finish(), "");
    }

    #[test]
    fn waits_for_multibyte_sequences() {
        let bytes = "你".as_bytes();
        let mut decoder = Utf8Decoder::new();
        assert_eq!(decoder.push(bytes[0]), None);
        assert_eq!(decoder.push(bytes[1]), None);
        assert_eq!(decoder.push(bytes[2]), Some("你".into()));
    }

    #[test]
    fn flushes_incomplete_tail_lossily() {
        let mut decoder = Utf8Decoder::new();
        assert_eq!(decoder.push(0xE4), None);
        assert_eq!(decoder.finish(), "\u{fffd}");
    }
}
